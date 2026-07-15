"""Resolve and classify Hub or local ASR model references."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_ASR_MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
DEFAULT_ASR_MODEL_REVISION = "0a8193caa4f3f92131471ab08824e488141cb392"

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_MODEL_WEIGHT_FILES = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
)
_ADAPTER_WEIGHT_FILES = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_model.bin",
    }
)

ModelFormat = Literal["dense", "bitsandbytes-int8", "bitsandbytes-int4"]


@dataclass(frozen=True, slots=True)
class ResolvedModelIdentity:
    """Immutable model and optional adapter identity used by one run."""

    model_id: str
    model_revision: str | None
    model_format: ModelFormat
    quantization_config: dict[str, Any] | None = None
    adapter_id: str | None = None
    adapter_revision: str | None = None


def default_model_revision(model_id: str, revision: str | None) -> str | None:
    """Apply the evaluated commit only to the package's default model."""
    if revision is None and model_id == DEFAULT_ASR_MODEL_ID:
        return DEFAULT_ASR_MODEL_REVISION
    return revision


def resolve_local_directory(reference: str, *, description: str) -> str | None:
    """Return a canonical local directory, or ``None`` for a possible Hub ID."""
    try:
        path = Path(reference).expanduser()
        if path.is_dir():
            return os.fspath(path.resolve(strict=True))
        path_exists = os.path.lexists(path)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Cannot resolve {description.lower()} path {reference!r}: {exc}"
        ) from exc
    if path_exists:
        raise ValueError(f"{description} path {reference!r} is not a directory")

    explicit_path = (
        path.is_absolute()
        or reference in {".", "..", "~"}
        or reference.startswith(("./", "../", "~/", "~"))
        or reference.count("/") > 1
    )
    if explicit_path:
        raise ValueError(f"{description} directory {reference!r} does not exist")
    return None


def _resolve_hub_revision(
    repo_id: str,
    revision: str | None,
    *,
    filename: str,
) -> str:
    """Resolve one Hub file to the immutable snapshot containing it."""
    requested = revision
    if requested is not None and _COMMIT_PATTERN.fullmatch(requested):
        return requested.lower()

    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=requested,
        )
    )
    for index in range(len(config_path.parts) - 2, -1, -1):
        if config_path.parts[index] != "snapshots":
            continue
        commit = config_path.parts[index + 1]
        if _COMMIT_PATTERN.fullmatch(commit):
            return commit.lower()
    raise RuntimeError(
        f"Cannot determine the immutable commit for {repo_id!r} from "
        f"Hugging Face cache path {config_path}"
    )


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    """Resolve a model commit, tag, branch, or default branch."""
    return _resolve_hub_revision(
        model_id,
        default_model_revision(model_id, revision),
        filename="config.json",
    )


def resolve_adapter_revision(adapter_id: str, revision: str | None) -> str:
    """Resolve an adapter commit, tag, branch, or default branch."""
    return _resolve_hub_revision(
        adapter_id,
        revision,
        filename="adapter_config.json",
    )


def _hub_json(repo_id: str, revision: str, filename: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{repo_id}@{revision}/{filename} is not a JSON object")
    return payload


def _local_json(directory: str, filename: str) -> dict[str, Any]:
    path = Path(directory, filename)
    if not path.is_file():
        raise ValueError(f"Local artifact {path} is missing or is not a file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _reference_json(
    reference: str, revision: str | None, filename: str
) -> dict[str, Any]:
    if revision is None:
        return _local_json(reference, filename)
    return _hub_json(reference, revision, filename)


def _verify_weight_artifacts(
    repo_id: str,
    revision: str | None,
    *,
    metadata_filename: str,
    candidates: frozenset[str],
    description: str,
) -> None:
    """Require loadable weights without downloading a full single-file model."""
    if revision is None:
        directory = Path(repo_id)
        if any((directory / name).is_file() for name in candidates):
            return
        expected = ", ".join(sorted(candidates))
        raise ValueError(
            f"Local directory {repo_id!r} does not contain supported {description}; "
            f"expected one of: {expected}"
        )

    from huggingface_hub import HfApi, hf_hub_download

    metadata_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=metadata_filename,
            revision=revision,
        )
    )
    # Normal repeated inference should stay cache-only and avoid Hub latency.
    if any((metadata_path.parent / name).is_file() for name in candidates):
        return
    repository_files = set(HfApi().list_repo_files(repo_id, revision=revision))
    if repository_files.isdisjoint(candidates):
        expected = ", ".join(sorted(candidates))
        raise ValueError(
            f"{repo_id}@{revision} does not contain supported {description}; "
            f"expected one of: {expected}"
        )


def classify_model_config(
    config: dict[str, Any], reference: str
) -> tuple[ModelFormat, dict[str, Any] | None]:
    if config.get("model_type") != "cohere_asr":
        raise ValueError(
            f"{reference} uses model_type={config.get('model_type')!r}; "
            "expected a native Transformers Cohere ASR checkpoint"
        )
    architectures = config.get("architectures")
    if architectures and "CohereAsrForConditionalGeneration" not in architectures:
        raise ValueError(
            f"{reference} does not declare CohereAsrForConditionalGeneration"
        )

    quantization = config.get("quantization_config")
    if quantization is None:
        return "dense", None
    if not isinstance(quantization, dict):
        raise ValueError(f"{reference} has an invalid quantization_config")
    method = str(quantization.get("quant_method", "")).lower()
    load_4bit = quantization.get(
        "load_in_4bit", quantization.get("_load_in_4bit", False)
    )
    load_8bit = quantization.get(
        "load_in_8bit", quantization.get("_load_in_8bit", False)
    )
    if not isinstance(load_4bit, bool) or not isinstance(load_8bit, bool):
        raise ValueError(
            f"{reference} has an invalid quantization_config: "
            "bitsandbytes load flags must be boolean"
        )
    if method != "bitsandbytes" or load_4bit == load_8bit:
        raise ValueError(
            f"{reference} uses unsupported saved quantization configuration "
            f"{method or 'unknown'!r}"
        )
    return (
        "bitsandbytes-int4" if load_4bit else "bitsandbytes-int8",
        dict(quantization),
    )


def resolve_model_identity(
    model_id: str,
    model_revision: str | None,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
    *,
    verify_weight_artifacts: bool = True,
) -> ResolvedModelIdentity:
    """Resolve model sources and reject unsupported formats early."""
    if adapter_id is None and adapter_revision is not None:
        raise ValueError("adapter_revision requires an adapter_id")

    local_model_id = resolve_local_directory(model_id, description="Model")
    if local_model_id is not None:
        if model_revision is not None:
            raise ValueError(
                "A model revision cannot be used with a local model directory"
            )
        resolved_model_id = local_model_id
        resolved_model_revision = None
    else:
        resolved_model_id = model_id
        resolved_model_revision = resolve_model_revision(model_id, model_revision)

    model_config = _reference_json(
        resolved_model_id, resolved_model_revision, "config.json"
    )
    reference = model_reference(resolved_model_id, resolved_model_revision)
    model_format, quantization = classify_model_config(model_config, reference)
    resolved_adapter_id = None
    resolved_adapter_revision = None
    if adapter_id is not None:
        if model_format != "dense":
            raise ValueError(
                "PEFT adapters are currently supported only with dense base models"
            )
        local_adapter_id = resolve_local_directory(adapter_id, description="Adapter")
        if local_adapter_id is not None:
            if adapter_revision is not None:
                raise ValueError(
                    "An adapter revision cannot be used with a local adapter directory"
                )
            resolved_adapter_id = local_adapter_id
        else:
            resolved_adapter_id = adapter_id
            resolved_adapter_revision = resolve_adapter_revision(
                adapter_id, adapter_revision
            )
        adapter_config = _reference_json(
            resolved_adapter_id,
            resolved_adapter_revision,
            "adapter_config.json",
        )
        if str(adapter_config.get("peft_type", "")).upper() != "LORA":
            raise ValueError("Only LoRA PEFT adapters are currently supported")
        if str(adapter_config.get("task_type", "")).upper() != "SEQ_2_SEQ_LM":
            raise ValueError("The PEFT adapter must declare task_type='SEQ_2_SEQ_LM'")
        if resolved_model_revision is not None:
            adapter_reference = model_reference(
                resolved_adapter_id, resolved_adapter_revision
            )
            adapter_base = adapter_config.get("base_model_name_or_path")
            if adapter_base != resolved_model_id:
                raise ValueError(
                    f"Adapter {adapter_reference} requires base model "
                    f"{adapter_base!r}, not {resolved_model_id!r}"
                )
            adapter_base_revision = adapter_config.get("revision")
            if adapter_base_revision is not None:
                expected_base_revision = resolve_model_revision(
                    resolved_model_id, str(adapter_base_revision)
                )
                if expected_base_revision != resolved_model_revision:
                    raise ValueError(
                        f"Adapter {adapter_reference} requires base revision "
                        f"{expected_base_revision}, not {resolved_model_revision}"
                    )
    identity = ResolvedModelIdentity(
        model_id=resolved_model_id,
        model_revision=resolved_model_revision,
        model_format=model_format,
        quantization_config=quantization,
        adapter_id=resolved_adapter_id,
        adapter_revision=resolved_adapter_revision,
    )
    if verify_weight_artifacts:
        verify_model_weight_artifacts(identity)
    return identity


def verify_model_weight_artifacts(identity: ResolvedModelIdentity) -> None:
    """Require weight entry points only when a run must load the ASR model."""
    _verify_weight_artifacts(
        identity.model_id,
        identity.model_revision,
        metadata_filename="config.json",
        candidates=_MODEL_WEIGHT_FILES,
        description="Transformers model weights",
    )
    if identity.adapter_id is not None:
        _verify_weight_artifacts(
            identity.adapter_id,
            identity.adapter_revision,
            metadata_filename="adapter_config.json",
            candidates=_ADAPTER_WEIGHT_FILES,
            description="PEFT adapter weights",
        )


def model_reference(model_id: str, revision: str | None) -> str:
    """Render a concise model reference for messages and telemetry."""
    return f"{model_id}@{revision}" if revision else model_id
