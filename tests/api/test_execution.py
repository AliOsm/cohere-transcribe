from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from cohere_transcribe import (
    PublicationOptions,
    TranscriptionConfigurationError,
    TranscriptionOptions,
    TranscriptionRun,
    TranscriptionRuntimeError,
    transcribe,
)
from cohere_transcribe.model_identity import ResolvedModelIdentity

from ._support import patch_execute, run_for


def test_publication_none_is_in_memory_and_publication_options_are_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bool, list[str], str | None, str, str | None]] = []

    def fake_execute(args, requested_options, *, publication_enabled, **_kwargs):
        seen.append(
            (
                publication_enabled,
                args.formats,
                args.output_dir,
                args.existing,
                args.profile_json,
            )
        )
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    transcribe("memory.wav")
    transcribe(
        "published.wav",
        options=TranscriptionOptions(
            publication=PublicationOptions(
                formats=("txt", "json"),
                output_dir=tmp_path / "out",
                existing="overwrite",
                profile_json=tmp_path / "profile.json",
            )
        ),
    )

    assert seen == [
        (False, ["txt", "srt", "vtt"], None, "error", None),
        (
            True,
            ["txt", "json"],
            os.fspath(tmp_path / "out"),
            "overwrite",
            os.fspath(tmp_path / "profile.json"),
        ),
    ]


def test_public_in_memory_execution_returns_text_without_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        assert not publish_outputs
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["captured text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=False)

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )

    run = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            alignment="segment",
        ),
    )

    assert run.single.status == "completed"
    assert run.single.text == "captured text"
    assert run.single.outputs == ()
    assert not run.single.provenance.published
    assert {path.name for path in tmp_path.iterdir()} == {"clip.wav"}


def test_selected_adapter_provenance_reaches_api_json_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    output_dir = tmp_path / "out"
    profile_path = tmp_path / "profile.json"
    model_revision = "1" * 40
    adapter_revision = "2" * 40
    verified: list[ResolvedModelIdentity] = []

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime,
        "resolve_model_identity",
        lambda *_args, **_kwargs: ResolvedModelIdentity(
            model_id="owner/model",
            model_revision=model_revision,
            model_format="dense",
            adapter_id="owner/adapter",
            adapter_revision=adapter_revision,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "verify_model_weight_artifacts",
        lambda identity: verified.append(identity),
    )
    monkeypatch.setattr(
        runtime, "preflight_runtime", lambda _args, _require_model_runtime: None
    )

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        assert publish_outputs
        for job in jobs:
            assert (job.model_id, job.model_revision, job.model_format) == (
                "owner/model",
                model_revision,
                "dense",
            )
            assert (job.adapter_id, job.adapter_revision) == (
                "owner/adapter",
                adapter_revision,
            )
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["selected model text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=True)

    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )
    options = TranscriptionOptions(
        model="owner/model",
        model_revision="release",
        adapter="owner/adapter",
        adapter_revision="adapter-release",
        vad="none",
        audio_backend="librosa",
        alignment="segment",
        publication=PublicationOptions(
            formats=("json",),
            output_dir=output_dir,
            existing="overwrite",
            profile_json=profile_path,
        ),
    )

    run = transcribe(source, options=options)

    assert len(verified) == 1
    assert run.single.provenance.model_id == "owner/model"
    assert run.single.provenance.model_revision == model_revision
    assert run.single.provenance.adapter_id == "owner/adapter"
    assert run.single.provenance.adapter_revision == adapter_revision
    output = json.loads((output_dir / "clip.json").read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected_adapter = {"id": "owner/adapter", "revision": adapter_revision}
    assert output["models"]["asr"]["revision"] == model_revision
    assert output["models"]["asr"]["adapter"] == expected_adapter
    assert profile["models"]["asr"]["revision"] == model_revision
    assert profile["models"]["asr"]["adapter"] == expected_adapter


def test_local_paths_are_canonical_with_null_revisions_in_all_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "cohere_asr"}), encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"fixture")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "training-machine/base",
                "peft_type": "LORA",
                "task_type": "SEQ_2_SEQ_LM",
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    output_dir = tmp_path / "out"
    profile_path = tmp_path / "profile.json"

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime, "preflight_runtime", lambda _args, _require_model_runtime: None
    )

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        assert args.model == str(model.resolve())
        assert args.model_revision is None
        assert args.adapter == str(adapter.resolve())
        assert args.adapter_revision is None
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["local model text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=publish_outputs)

    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )
    options = TranscriptionOptions(
        model=model,
        adapter=adapter,
        vad="none",
        audio_backend="librosa",
        alignment="segment",
        publication=PublicationOptions(
            formats=("json",),
            output_dir=output_dir,
            existing="overwrite",
            profile_json=profile_path,
        ),
    )

    run = transcribe(source, options=options)

    expected_adapter = {"id": str(adapter.resolve()), "revision": None}
    output = json.loads((output_dir / "clip.json").read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert run.requested_options.model == model
    assert run.resolved_options.model == str(model.resolve())
    assert run.resolved_options.model_revision is None
    assert run.single.provenance.model_id == str(model.resolve())
    assert run.single.provenance.model_revision is None
    assert run.single.provenance.adapter_id == str(adapter.resolve())
    assert run.single.provenance.adapter_revision is None
    assert output["models"]["asr"] == {
        "id": str(model.resolve()),
        "revision": None,
        "format": "dense",
        "quantization": None,
        "adapter": expected_adapter,
    }
    assert profile["models"]["asr"]["id"] == str(model.resolve())
    assert profile["models"]["asr"]["revision"] is None
    assert profile["models"]["asr"]["adapter"] == expected_adapter


def test_publication_writes_outputs_and_verified_skip_does_not_run_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    output_dir = tmp_path / "out"
    calls = 0

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        nonlocal calls
        calls += 1
        assert publish_outputs
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["published text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=True)

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )
    base_publication = PublicationOptions(
        formats=("txt", "json"),
        output_dir=output_dir,
        existing="overwrite",
    )
    first = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            publication=base_publication,
        ),
    )
    second = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            publication=replace(base_publication, existing="skip"),
        ),
    )

    assert calls == 1
    assert first.single.status == "completed"
    assert first.single.provenance.published
    assert {path.name for path in first.single.outputs} == {"clip.txt", "clip.json"}
    assert second.single.status == "skipped"
    assert second.single.provenance.published
    assert {path.name for path in second.single.outputs} == {"clip.txt", "clip.json"}


def test_configuration_system_exit_is_exposed_as_a_typed_api_error() -> None:
    invalid = TranscriptionOptions(language=cast(Any, "fr"))
    with pytest.raises(TranscriptionConfigurationError, match="language"):
        transcribe("input.wav", options=invalid)


def test_runtime_system_exit_is_exposed_as_a_typed_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_execute(*_args, **_kwargs):
        raise SystemExit("backend initialization failed")

    patch_execute(monkeypatch, fake_execute)
    with pytest.raises(
        TranscriptionRuntimeError, match="backend initialization failed"
    ):
        transcribe("input.wav")


def test_invalid_publication_object_is_exposed_as_a_configuration_error() -> None:
    options = TranscriptionOptions(publication=cast(Any, object()))
    with pytest.raises(TranscriptionConfigurationError):
        transcribe("input.wav", options=options)


def test_in_memory_api_does_not_initialize_filesystem_output_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    def fail_output_mode() -> int:
        raise AssertionError("in-memory API must not initialize publication state")

    def fake_execute(
        _args, requested_options, *, publication_enabled: bool, **_kwargs
    ) -> TranscriptionRun:
        assert not publication_enabled
        return run_for(requested_options)

    monkeypatch.setattr(runtime, "default_output_mode", fail_output_mode)
    monkeypatch.setattr(runtime, "execute", fake_execute)

    assert transcribe("memory.wav").ok
