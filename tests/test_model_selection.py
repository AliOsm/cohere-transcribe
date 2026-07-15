"""Model identity, compatibility, and reusable-resource contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cohere_transcribe import TranscriptionOptions
from cohere_transcribe.asr.model import (
    MemoizedEncoderProjection,
    load_asr,
    load_asr_config_and_processor,
)
from cohere_transcribe.config import config_from_options, parse_args, validate_args
from cohere_transcribe.model_identity import (
    _MODEL_WEIGHT_FILES,
    DEFAULT_ASR_MODEL_ID,
    DEFAULT_ASR_MODEL_REVISION,
    _verify_weight_artifacts,
    classify_model_config,
    resolve_model_identity,
    resolve_model_revision,
    verify_model_weight_artifacts,
)
from cohere_transcribe.preflight import preflight_runtime
from cohere_transcribe.runtime.resources import ModelResources


def allow_test_weight_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._verify_weight_artifacts",
        lambda *_args, **_kwargs: None,
    )


def write_local_model(
    directory: Path, *, quantization: dict[str, object] | None = None
) -> Path:
    directory.mkdir(parents=True)
    config: dict[str, object] = {
        "model_type": "cohere_asr",
        "architectures": ["CohereAsrForConditionalGeneration"],
    }
    if quantization is not None:
        config["quantization_config"] = quantization
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    return directory


def write_local_adapter(directory: Path, *, base: str = "owner/model") -> Path:
    directory.mkdir(parents=True)
    config = {
        "base_model_name_or_path": base,
        "peft_type": "LORA",
        "task_type": "SEQ_2_SEQ_LM",
    }
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"fixture")
    return directory


def fail_hub_call(*_args, **_kwargs):
    raise AssertionError("local model resolution must not contact the Hub")


def test_cli_accepts_model_and_adapter_revisions() -> None:
    args = parse_args(
        [
            "audio.wav",
            "--model",
            "owner/model",
            "--model-revision",
            "release",
            "--adapter",
            "owner/adapter",
            "--adapter-revision",
            "adapter-release",
        ]
    )
    validate_args(args)

    assert args.model == "owner/model"
    assert args.model_revision == "release"
    assert args.adapter == "owner/adapter"
    assert args.adapter_revision == "adapter-release"


def test_cli_rejects_adapter_revision_without_adapter() -> None:
    args = parse_args(["audio.wav", "--adapter-revision", "release"])

    with pytest.raises(SystemExit, match="requires --adapter"):
        validate_args(args)


def test_cli_accepts_existing_local_model_and_adapter_directories(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    args = parse_args(["audio.wav", "--model", str(model), "--adapter", str(adapter)])

    validate_args(args)

    assert args.model == str(model)
    assert args.model_revision is None
    assert args.adapter == str(adapter)
    assert args.adapter_revision is None


def test_python_options_accept_path_like_model_and_adapter(tmp_path: Path) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    options = TranscriptionOptions(model=model, adapter=adapter)

    args = config_from_options(["audio.wav"], options)
    validate_args(args)

    assert args.model == str(model)
    assert args.adapter == str(adapter)


@pytest.mark.parametrize(
    ("flag", "revision_flag", "directory_name", "message"),
    [
        (
            "--model",
            "--model-revision",
            "model",
            "model-revision cannot be used with a local model",
        ),
        (
            "--adapter",
            "--adapter-revision",
            "adapter",
            "adapter-revision cannot be used with a local adapter",
        ),
    ],
)
def test_cli_rejects_revisions_for_local_directories(
    tmp_path: Path,
    flag: str,
    revision_flag: str,
    directory_name: str,
    message: str,
) -> None:
    directory = tmp_path / directory_name
    directory.mkdir()
    args = parse_args(["audio.wav", flag, str(directory), revision_flag, "candidate"])

    with pytest.raises(SystemExit, match=message):
        validate_args(args)


@pytest.mark.parametrize("reference", ["./missing", "../missing", "~/missing"])
def test_cli_rejects_missing_explicit_local_paths(reference: str) -> None:
    args = parse_args(["audio.wav", "--model", reference])

    with pytest.raises(SystemExit, match=r"directory .* does not exist"):
        validate_args(args)


def test_cli_rejects_an_existing_file_as_a_model_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"fixture")
    args = parse_args(["audio.wav", "--model", str(file_path)])

    with pytest.raises(SystemExit, match="is not a directory"):
        validate_args(args)


def test_cli_normalizes_an_unresolvable_tilde_path_to_a_configuration_error() -> None:
    args = parse_args(["audio.wav", "--model", "~definitely-no-such-user-ct/model"])

    with pytest.raises(SystemExit, match="Cannot resolve model path"):
        validate_args(args)


def test_preflight_imports_quantized_extra(monkeypatch) -> None:
    args = parse_args(
        [
            "audio.wav",
            "--vad",
            "none",
            "--alignment",
            "none",
            "--audio-backend",
            "librosa",
        ]
    )
    validate_args(args)
    args.device = "cuda"
    args.model_format = "bitsandbytes-int8"
    imported: list[str] = []
    versions = {
        "transformers": "5.13.1",
        "bitsandbytes": "0.49.2",
        "accelerate": "1.13.0",
    }
    monkeypatch.setattr("cohere_transcribe.preflight.package_version", versions.get)
    monkeypatch.setattr(
        "cohere_transcribe.preflight.importlib.import_module",
        lambda name: imported.append(name) or object(),
    )

    preflight_runtime(args)

    assert imported == [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "librosa",
    ]


def test_preflight_imports_adapter_extra(monkeypatch) -> None:
    args = parse_args(
        [
            "audio.wav",
            "--vad",
            "none",
            "--alignment",
            "none",
            "--audio-backend",
            "librosa",
            "--adapter",
            "owner/adapter",
        ]
    )
    validate_args(args)
    args.device = "cpu"
    args.model_format = "dense"
    imported: list[str] = []
    versions = {"transformers": "5.13.1", "peft": "0.19.1"}
    monkeypatch.setattr("cohere_transcribe.preflight.package_version", versions.get)
    monkeypatch.setattr(
        "cohere_transcribe.preflight.importlib.import_module",
        lambda name: imported.append(name) or object(),
    )

    preflight_runtime(args)

    assert imported == ["transformers", "peft", "librosa"]


@pytest.mark.parametrize(
    ("quantization", "expected_format"),
    [
        (None, "dense"),
        (
            {"quant_method": "bitsandbytes", "load_in_8bit": True},
            "bitsandbytes-int8",
        ),
        (
            {"quant_method": "bitsandbytes", "load_in_4bit": True},
            "bitsandbytes-int4",
        ),
    ],
)
def test_local_model_identity_is_canonical_and_never_contacts_the_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization: dict[str, object] | None,
    expected_format: str,
) -> None:
    from huggingface_hub import HfApi

    model = write_local_model(tmp_path / "model", quantization=quantization)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_hub_call)
    monkeypatch.setattr(HfApi, "list_repo_files", fail_hub_call)

    identity = resolve_model_identity(str(model), None)

    assert identity.model_id == str(model.resolve())
    assert identity.model_revision is None
    assert identity.model_format == expected_format
    assert identity.quantization_config == quantization


def test_relative_and_symlinked_local_paths_share_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_local_model(tmp_path / "model")
    link = tmp_path / "model-link"
    link.symlink_to(model, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    relative = resolve_model_identity("model", None)
    linked = resolve_model_identity("model-link", None)

    assert relative == linked
    assert relative.model_id == str(model.resolve())


def test_local_identity_rejects_a_model_revision(tmp_path: Path) -> None:
    model = write_local_model(tmp_path / "model")

    with pytest.raises(ValueError, match="revision cannot be used with a local model"):
        resolve_model_identity(str(model), "candidate")


def test_fully_local_adapter_identity_and_weights_never_contact_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import HfApi

    model = write_local_model(tmp_path / "model")
    adapter = write_local_adapter(tmp_path / "adapter", base="training-machine/base")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_hub_call)
    monkeypatch.setattr(HfApi, "list_repo_files", fail_hub_call)

    identity = resolve_model_identity(str(model), None, str(adapter), None)

    assert identity.model_id == str(model.resolve())
    assert identity.model_revision is None
    assert identity.adapter_id == str(adapter.resolve())
    assert identity.adapter_revision is None


def test_local_identity_rejects_an_adapter_revision(tmp_path: Path) -> None:
    model = write_local_model(tmp_path / "model")
    adapter = write_local_adapter(tmp_path / "adapter")

    with pytest.raises(
        ValueError, match="revision cannot be used with a local adapter"
    ):
        resolve_model_identity(str(model), None, str(adapter), "candidate")


def test_hub_base_still_validates_a_local_adapters_declared_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = write_local_adapter(tmp_path / "adapter", base="owner/other")
    allow_test_weight_artifacts(monkeypatch)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision",
        lambda *_args: "1" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda *_args: {"model_type": "cohere_asr"},
    )

    with pytest.raises(ValueError, match="requires base model"):
        resolve_model_identity("owner/model", None, str(adapter), None)


def test_local_base_does_not_compare_a_hub_adapters_historical_base_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = write_local_model(tmp_path / "model")
    allow_test_weight_artifacts(monkeypatch)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_adapter_revision",
        lambda *_args: "2" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda *_args: {
            "base_model_name_or_path": "training-machine/base",
            "peft_type": "LORA",
            "task_type": "SEQ_2_SEQ_LM",
        },
    )

    identity = resolve_model_identity(str(model), None, "owner/adapter", "2" * 40)

    assert identity.adapter_id == "owner/adapter"
    assert identity.adapter_revision == "2" * 40


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "config.json.*missing"),
        ("not-json", "Cannot read JSON object"),
        ("[]", "not a JSON object"),
    ],
)
def test_local_identity_rejects_invalid_configuration(
    tmp_path: Path, contents: str | None, message: str
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture")
    if contents is not None:
        (model / "config.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        resolve_model_identity(str(model), None)


def test_local_weight_validation_can_be_deferred(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "cohere_asr"}), encoding="utf-8"
    )

    identity = resolve_model_identity(str(model), None, verify_weight_artifacts=False)

    with pytest.raises(ValueError, match="Transformers model weights"):
        verify_model_weight_artifacts(identity)


def test_local_adapter_weight_validation_is_not_skipped(tmp_path: Path) -> None:
    model = write_local_model(tmp_path / "model")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "SEQ_2_SEQ_LM",
                "base_model_name_or_path": "unused/local-base",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PEFT adapter weights"):
        resolve_model_identity(str(model), None, str(adapter), None)


def test_default_and_explicit_commits_resolve_without_hub_lookup(monkeypatch) -> None:
    def fail_download(**_kwargs):
        raise AssertionError("an immutable revision must not contact the Hub")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_download)

    assert resolve_model_revision(DEFAULT_ASR_MODEL_ID, None) == (
        DEFAULT_ASR_MODEL_REVISION
    )
    custom_commit = "A" * 40
    assert resolve_model_revision("owner/model", custom_commit) == custom_commit.lower()


@pytest.mark.parametrize(
    "revision",
    [None, DEFAULT_ASR_MODEL_REVISION, DEFAULT_ASR_MODEL_REVISION.upper()],
)
def test_packaged_default_identity_and_weights_need_no_hub_access(
    monkeypatch, revision: str | None
) -> None:
    from huggingface_hub import HfApi

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_hub_call)
    monkeypatch.setattr(HfApi, "list_repo_files", fail_hub_call)

    identity = resolve_model_identity(DEFAULT_ASR_MODEL_ID, revision)
    verify_model_weight_artifacts(identity)

    assert identity.model_id == DEFAULT_ASR_MODEL_ID
    assert identity.model_revision == DEFAULT_ASR_MODEL_REVISION
    assert identity.model_format == "dense"
    assert identity.quantization_config is None


def test_another_default_repository_revision_still_inspects_its_config(
    monkeypatch,
) -> None:
    revision = "1" * 40
    inspected: list[tuple[str, str | None, str]] = []
    allow_test_weight_artifacts(monkeypatch)

    def inspect(reference: str, resolved_revision: str | None, filename: str):
        inspected.append((reference, resolved_revision, filename))
        return {
            "model_type": "cohere_asr",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_8bit": True,
            },
        }

    monkeypatch.setattr("cohere_transcribe.model_identity._reference_json", inspect)

    identity = resolve_model_identity(DEFAULT_ASR_MODEL_ID, revision)

    assert identity.model_revision == revision
    assert identity.model_format == "bitsandbytes-int8"
    assert inspected == [(DEFAULT_ASR_MODEL_ID, revision, "config.json")]


def test_packaged_default_adapter_metadata_and_weights_are_still_validated(
    monkeypatch,
) -> None:
    adapter_revision = "2" * 40
    inspected: list[tuple[str, str | None, str]] = []
    verified: list[tuple[str, str | None, str]] = []
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_adapter_revision",
        lambda *_args: adapter_revision,
    )

    def inspect(reference: str, revision: str | None, filename: str):
        inspected.append((reference, revision, filename))
        return {
            "base_model_name_or_path": DEFAULT_ASR_MODEL_ID,
            "peft_type": "LORA",
            "task_type": "SEQ_2_SEQ_LM",
        }

    def verify(reference: str, revision: str | None, *, description: str, **_kwargs):
        verified.append((reference, revision, description))

    monkeypatch.setattr("cohere_transcribe.model_identity._reference_json", inspect)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._verify_weight_artifacts", verify
    )

    identity = resolve_model_identity(
        DEFAULT_ASR_MODEL_ID,
        None,
        "owner/adapter",
        adapter_revision,
    )

    assert identity.adapter_revision == adapter_revision
    assert inspected == [
        ("owner/adapter", adapter_revision, "adapter_config.json"),
    ]
    assert verified == [
        ("owner/adapter", adapter_revision, "PEFT adapter weights"),
    ]


def test_symbolic_revision_resolves_from_the_snapshot_cache_path(
    tmp_path, monkeypatch
) -> None:
    commit = "1" * 40
    config = tmp_path / "models--owner--model" / "snapshots" / commit / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(config)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    assert resolve_model_revision("owner/model", "release") == commit
    assert calls == [
        {
            "repo_id": "owner/model",
            "filename": "config.json",
            "revision": "release",
        }
    ]


def test_symbolic_revision_uses_the_innermost_valid_snapshot_directory(
    tmp_path, monkeypatch
) -> None:
    commit = "2" * 40
    config = (
        tmp_path
        / "snapshots"
        / "unrelated-cache-root"
        / "models--owner--model"
        / "snapshots"
        / commit
        / "config.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: str(config)
    )

    assert resolve_model_revision("owner/model", "release") == commit


def test_weight_artifact_validation_accepts_transformers_weights(
    tmp_path, monkeypatch
) -> None:
    from huggingface_hub import HfApi

    config = tmp_path / "snapshots" / ("1" * 40) / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: str(config)
    )
    monkeypatch.setattr(
        HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["config.json", "model.safetensors.index.json"],
    )

    _verify_weight_artifacts(
        "owner/model",
        "1" * 40,
        metadata_filename="config.json",
        candidates=_MODEL_WEIGHT_FILES,
        description="Transformers model weights",
    )


def test_weight_artifact_validation_rejects_onnx_only_repository(
    tmp_path, monkeypatch
) -> None:
    from huggingface_hub import HfApi

    config = tmp_path / "snapshots" / ("1" * 40) / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: str(config)
    )
    monkeypatch.setattr(
        HfApi,
        "list_repo_files",
        lambda *_args, **_kwargs: ["config.json", "model.onnx"],
    )

    with pytest.raises(ValueError, match="Transformers model weights"):
        _verify_weight_artifacts(
            "owner/onnx",
            "1" * 40,
            metadata_filename="config.json",
            candidates=_MODEL_WEIGHT_FILES,
            description="Transformers model weights",
        )


def test_weight_artifact_validation_uses_cached_snapshot_offline(
    tmp_path, monkeypatch
) -> None:
    from huggingface_hub import HfApi

    snapshot = tmp_path / "snapshots" / ("1" * 40)
    snapshot.mkdir(parents=True)
    config = snapshot / "config.json"
    config.write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"cached")
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda **_kwargs: str(config)
    )

    def unexpected_api_call(*_args, **_kwargs):
        raise AssertionError("cached weight validation must not contact the Hub API")

    monkeypatch.setattr(HfApi, "list_repo_files", unexpected_api_call)

    _verify_weight_artifacts(
        "owner/model",
        "1" * 40,
        metadata_filename="config.json",
        candidates=_MODEL_WEIGHT_FILES,
        description="Transformers model weights",
    )


def test_model_identity_can_defer_weight_validation_until_inference(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision",
        lambda *_args: "1" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda *_args: {"model_type": "cohere_asr"},
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._verify_weight_artifacts",
        lambda repo_id, revision, **_kwargs: calls.append((repo_id, revision)),
    )

    identity = resolve_model_identity(
        "owner/model", None, verify_weight_artifacts=False
    )
    assert calls == []

    verify_model_weight_artifacts(identity)
    assert calls == [("owner/model", "1" * 40)]


def test_checkpoint_only_preflight_does_not_require_inference_dependencies(
    monkeypatch,
) -> None:
    args = parse_args(["audio.wav"])
    validate_args(args)
    args.device = "cpu"
    args.model_format = "bitsandbytes-int8"
    args.adapter = "owner/adapter"
    imported: list[str] = []
    monkeypatch.setattr(
        "cohere_transcribe.preflight.package_version",
        lambda name: pytest.fail(f"unexpected version lookup for {name}"),
    )
    monkeypatch.setattr(
        "cohere_transcribe.preflight.importlib.import_module",
        lambda name: imported.append(name) or object(),
    )

    preflight_runtime(args, require_model_runtime=False)

    assert imported == []


def test_checkpoint_only_execution_skips_weight_artifact_validation(
    tmp_path, monkeypatch
) -> None:
    from dataclasses import asdict
    from pathlib import Path

    from cohere_transcribe import TranscriptionOptions
    from cohere_transcribe.config import config_from_options
    from cohere_transcribe.model_identity import ResolvedModelIdentity
    from cohere_transcribe.models import AudioJob, SourceSnapshot
    from cohere_transcribe.runtime import engine

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    options = TranscriptionOptions(
        model="owner/int8",
        vad="none",
        alignment="segment",
        audio_backend="librosa",
    )
    args = config_from_options([str(source)], options)
    validate_args(args)
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="segment",
        model_id="owner/int8",
        model_revision="3" * 40,
        model_format="bitsandbytes-int8",
        asr_checkpoint_loaded=True,
        duration=1.0,
        segment_times=[(0.0, 1.0)],
        segment_texts=["checkpoint text"],
    )
    preflight_modes: list[bool] = []
    monkeypatch.setattr(
        engine,
        "_resolve_precision",
        lambda config: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            config.dtype,
            config.vad_engine,
        ),
    )
    monkeypatch.setattr(
        engine,
        "resolve_model_identity",
        lambda *_args, **_kwargs: ResolvedModelIdentity(
            model_id="owner/int8",
            model_revision="3" * 40,
            model_format="bitsandbytes-int8",
        ),
    )
    monkeypatch.setattr(
        engine.inputs_module,
        "build_jobs",
        lambda *_args, **_kwargs: [job],
    )
    monkeypatch.setattr(
        engine,
        "verify_model_weight_artifacts",
        lambda _identity: pytest.fail("checkpoint-only execution loaded weights"),
    )
    monkeypatch.setattr(
        engine.transcription_pipeline,
        "transcribe_all",
        lambda jobs, *_args, **_kwargs: (
            None
            if jobs == [job] and job.asr_checkpoint_loaded
            else pytest.fail("unexpected transcription jobs")
        ),
    )

    run = engine.execute(
        args,
        options,
        requested_configuration=asdict(args),
        resources=None,
        publication_enabled=False,
        console=False,
        preflight=lambda _args, required: preflight_modes.append(required),
    )

    assert preflight_modes == [False]
    assert run.single.text == "checkpoint text"
    assert run.single.provenance.model_format == "bitsandbytes-int8"


def test_native_component_loader_disables_remote_code_and_checks_types(
    monkeypatch,
) -> None:
    import transformers

    config = transformers.CohereAsrConfig()
    processor = object.__new__(transformers.CohereAsrProcessor)
    config_calls: list[tuple[str, dict[str, object]]] = []
    processor_calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_id, **kwargs: config_calls.append((model_id, kwargs)) or config,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda model_id, **kwargs: (
            processor_calls.append((model_id, kwargs)) or processor
        ),
    )

    loaded = load_asr_config_and_processor("owner/model", "2" * 40)

    assert loaded == (config, processor)
    expected = {"revision": "2" * 40, "trust_remote_code": False}
    assert config_calls == [("owner/model", expected)]
    assert processor_calls == [("owner/model", expected)]


def test_native_component_loader_omits_revision_for_a_local_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    model = tmp_path / "model"
    model.mkdir()
    config = transformers.CohereAsrConfig()
    processor = object.__new__(transformers.CohereAsrProcessor)
    config_calls: list[tuple[str, dict[str, object]]] = []
    processor_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_id, **kwargs: config_calls.append((model_id, kwargs)) or config,
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda model_id, **kwargs: (
            processor_calls.append((model_id, kwargs)) or processor
        ),
    )

    loaded = load_asr_config_and_processor(str(model), None)

    assert loaded == (config, processor)
    expected = {"trust_remote_code": False}
    assert config_calls == [(str(model), expected)]
    assert processor_calls == [(str(model), expected)]


def test_native_component_loader_rejects_an_unrelated_architecture(monkeypatch) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: transformers.WhisperConfig(),
    )

    with pytest.raises(ValueError, match="native Transformers Cohere ASR"):
        load_asr_config_and_processor("owner/model", "3" * 40)


def test_dense_loader_applies_hot_path_to_a_selected_model(monkeypatch) -> None:
    import transformers

    processor = object()
    configuration = SimpleNamespace(
        to_dict=lambda: {
            "model_type": "cohere_asr",
            "architectures": ["CohereAsrForConditionalGeneration"],
        }
    )

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(
                decoder=SimpleNamespace(proj=torch.nn.Linear(2, 2)),
                encoder=torch.nn.Identity(),
            )

    model = DummyModel()
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "cohere_transcribe.asr.model.load_asr_config_and_processor",
        lambda model_id, revision: (configuration, processor),
    )
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        lambda model_id, **kwargs: calls.append((model_id, kwargs)) or model,
    )

    loaded_processor, loaded_model = load_asr(
        "cpu", torch.float32, "owner/model", "4" * 40
    )

    assert loaded_processor is processor
    assert loaded_model is model
    assert not model.training
    assert isinstance(model.model.decoder.proj, MemoizedEncoderProjection)
    assert len(model.model.encoder._forward_hooks) == 1
    assert calls == [
        (
            "owner/model",
            {
                "config": configuration,
                "dtype": torch.float32,
                "attn_implementation": "sdpa",
                "revision": "4" * 40,
                "trust_remote_code": False,
            },
        )
    ]


def test_dense_loader_omits_revision_for_a_local_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    processor = object()
    configuration = SimpleNamespace(to_dict=lambda: {"model_type": "cohere_asr"})

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(
                decoder=SimpleNamespace(proj=torch.nn.Linear(2, 2)),
                encoder=torch.nn.Identity(),
            )

    model = DummyModel()
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "cohere_transcribe.asr.model.load_asr_config_and_processor",
        lambda *_args: (configuration, processor),
    )
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        lambda model_id, **kwargs: calls.append((model_id, kwargs)) or model,
    )

    load_asr("cpu", torch.float32, str(model_path), None)

    assert calls[0][0] == str(model_path)
    assert "revision" not in calls[0][1]


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [
        (None, "dense"),
        (
            {"quant_method": "bitsandbytes", "load_in_8bit": True},
            "bitsandbytes-int8",
        ),
        (
            {"quant_method": "bitsandbytes", "_load_in_4bit": True},
            "bitsandbytes-int4",
        ),
    ],
)
def test_saved_model_format_classification(quantization, expected) -> None:
    config = {
        "model_type": "cohere_asr",
        "architectures": ["CohereAsrForConditionalGeneration"],
    }
    if quantization is not None:
        config["quantization_config"] = quantization

    model_format, retained = classify_model_config(config, "owner/model@commit")

    assert model_format == expected
    assert retained == quantization


@pytest.mark.parametrize(
    "quantization",
    [
        "int8",
        {"quant_method": "gptq", "bits": 4},
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "load_in_8bit": True,
        },
        {"quant_method": "bitsandbytes"},
        {"quant_method": "bitsandbytes", "load_in_4bit": "false"},
    ],
)
def test_saved_model_format_rejects_unsupported_quantization(quantization) -> None:
    config = {
        "model_type": "cohere_asr",
        "architectures": ["CohereAsrForConditionalGeneration"],
        "quantization_config": quantization,
    }

    with pytest.raises(ValueError, match=r"quantization_config|quantization"):
        classify_model_config(config, "owner/model@commit")


def test_model_identity_resolves_and_validates_a_lora_adapter(monkeypatch) -> None:
    allow_test_weight_artifacts(monkeypatch)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision",
        lambda *_args: "1" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_adapter_revision",
        lambda *_args: "2" * 40,
    )

    def payload(repo_id, _revision, filename):
        if filename == "config.json":
            return {"model_type": "cohere_asr"}
        assert repo_id == "owner/adapter"
        return {
            "base_model_name_or_path": "owner/model",
            "peft_type": "LORA",
            "task_type": "SEQ_2_SEQ_LM",
        }

    monkeypatch.setattr("cohere_transcribe.model_identity._hub_json", payload)

    identity = resolve_model_identity("owner/model", None, "owner/adapter", "release")

    assert identity.model_format == "dense"
    assert identity.adapter_id == "owner/adapter"
    assert identity.adapter_revision == "2" * 40


def test_model_identity_rejects_adapter_base_mismatch(monkeypatch) -> None:
    allow_test_weight_artifacts(monkeypatch)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision",
        lambda *_args: "1" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_adapter_revision",
        lambda *_args: "2" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda _repo, _revision, filename: (
            {"model_type": "cohere_asr"}
            if filename == "config.json"
            else {
                "base_model_name_or_path": "owner/other",
                "peft_type": "LORA",
                "task_type": "SEQ_2_SEQ_LM",
            }
        ),
    )

    with pytest.raises(ValueError, match="requires base model"):
        resolve_model_identity("owner/model", None, "owner/adapter", None)


def test_model_identity_rejects_adapter_on_a_saved_quantized_model(
    monkeypatch,
) -> None:
    allow_test_weight_artifacts(monkeypatch)
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision",
        lambda *_args: "1" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda *_args: {
            "model_type": "cohere_asr",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_8bit": True,
            },
        },
    )

    with pytest.raises(ValueError, match="only with dense base models"):
        resolve_model_identity("owner/int8", None, "owner/adapter", None)


def test_model_identity_rejects_adapter_base_revision_mismatch(monkeypatch) -> None:
    allow_test_weight_artifacts(monkeypatch)
    requested_revision = "1" * 40
    adapter_base_revision = "3" * 40

    def resolve_revision(_model_id, revision):
        return (
            adapter_base_revision if revision == "adapter-base" else requested_revision
        )

    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_model_revision", resolve_revision
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity.resolve_adapter_revision",
        lambda *_args: "2" * 40,
    )
    monkeypatch.setattr(
        "cohere_transcribe.model_identity._hub_json",
        lambda _repo, _revision, filename: (
            {"model_type": "cohere_asr"}
            if filename == "config.json"
            else {
                "base_model_name_or_path": "owner/model",
                "revision": "adapter-base",
                "peft_type": "LORA",
                "task_type": "SEQ_2_SEQ_LM",
            }
        ),
    )

    with pytest.raises(ValueError, match="requires base revision"):
        resolve_model_identity("owner/model", None, "owner/adapter", None)


def test_quantized_loader_uses_device_map_and_never_moves_the_model(
    monkeypatch,
) -> None:
    import transformers

    processor = object()
    configuration = SimpleNamespace(
        to_dict=lambda: {
            "model_type": "cohere_asr",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_8bit": True,
            },
        }
    )

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(
                decoder=SimpleNamespace(proj=torch.nn.Linear(2, 2)),
                encoder=torch.nn.Identity(),
            )

        def to(self, *_args, **_kwargs):
            raise AssertionError("saved quantized models must not be moved after load")

    model = DummyModel()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "cohere_transcribe.asr.model.load_asr_config_and_processor",
        lambda *_args: (configuration, processor),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        lambda _model_id, **kwargs: calls.append(kwargs) or model,
    )

    loaded_processor, loaded_model = load_asr(
        "cuda",
        torch.bfloat16,
        "owner/int8",
        "3" * 40,
        "bitsandbytes-int8",
    )

    assert loaded_processor is processor
    assert loaded_model is model
    assert calls[0]["device_map"] == {"": "cuda:0"}
    assert isinstance(model.model.decoder.proj, MemoizedEncoderProjection)


def test_adapter_is_safely_merged_before_hot_path_patching(monkeypatch) -> None:
    import transformers

    processor = object()
    configuration = SimpleNamespace(to_dict=lambda: {"model_type": "cohere_asr"})

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(
                decoder=SimpleNamespace(proj=torch.nn.Linear(2, 2)),
                encoder=torch.nn.Identity(),
            )

    model = DummyModel()
    events: list[object] = []

    class AdapterModel:
        def merge_and_unload(self, *, safe_merge):
            events.append(("merge", safe_merge))
            assert not isinstance(model.model.decoder.proj, MemoizedEncoderProjection)
            return model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter_id, **kwargs):
            events.append(("load", base, adapter_id, kwargs))
            return AdapterModel()

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setattr(
        "cohere_transcribe.asr.model.load_asr_config_and_processor",
        lambda *_args: (configuration, processor),
    )
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    _, loaded = load_asr(
        "cpu",
        torch.float32,
        "owner/model",
        "4" * 40,
        "dense",
        "owner/adapter",
        "5" * 40,
    )

    assert loaded is model
    assert events[0][0] == "load"
    assert events[0][3] == {
        "revision": "5" * 40,
        "is_trainable": False,
        "low_cpu_mem_usage": True,
    }
    assert events[1] == ("merge", True)
    assert isinstance(model.model.decoder.proj, MemoizedEncoderProjection)


def test_local_adapter_loader_omits_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformers

    processor = object()
    configuration = SimpleNamespace(to_dict=lambda: {"model_type": "cohere_asr"})

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(
                decoder=SimpleNamespace(proj=torch.nn.Linear(2, 2)),
                encoder=torch.nn.Identity(),
            )

    model = DummyModel()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []

    class AdapterModel:
        def merge_and_unload(self, *, safe_merge):
            assert safe_merge
            return model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(_base, adapter_id, **kwargs):
            calls.append((adapter_id, kwargs))
            return AdapterModel()

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    monkeypatch.setattr(
        "cohere_transcribe.asr.model.load_asr_config_and_processor",
        lambda *_args: (configuration, processor),
    )
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    load_asr(
        "cpu",
        torch.float32,
        str(tmp_path / "model"),
        None,
        "dense",
        str(adapter),
        None,
    )

    assert calls == [
        (
            str(adapter),
            {"is_trainable": False, "low_cpu_mem_usage": True},
        )
    ]


def test_resource_cache_includes_the_model_identity() -> None:
    loads = 0

    def loader(_device: str, _dtype: torch.dtype):
        nonlocal loads
        loads += 1
        return object(), object()

    resources = ModelResources()
    try:
        first = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="owner/first",
            model_revision="5" * 40,
            loader=loader,
        )
        same = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="owner/first",
            model_revision="5" * 40,
            loader=loader,
        )
        changed = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="owner/second",
            model_revision="6" * 40,
            loader=loader,
        )
        adapted = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="owner/second",
            model_revision="6" * 40,
            adapter_id="owner/adapter",
            adapter_revision="7" * 40,
            loader=loader,
        )
    finally:
        resources.close()

    assert first[1] is same[1]
    assert changed[1] is not first[1]
    assert adapted[1] is not changed[1]
    assert loads == 3


def test_resource_cache_accepts_a_local_identity_without_revision() -> None:
    loads = 0

    def loader(_device: str, _dtype: torch.dtype):
        nonlocal loads
        loads += 1
        return object(), object()

    resources = ModelResources()
    try:
        first = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="/models/first",
            model_revision=None,
            loader=loader,
        )
        same = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="/models/first",
            model_revision=None,
            loader=loader,
        )
        changed = resources.acquire_asr(
            "cpu",
            torch.float32,
            model_id="/models/second",
            model_revision=None,
            loader=loader,
        )
    finally:
        resources.close()

    assert first[1] is same[1]
    assert changed[1] is not first[1]
    assert loads == 2
