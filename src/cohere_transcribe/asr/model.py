"""Cohere ASR model loading and model-specific inference optimizations."""

from __future__ import annotations

import torch

from ..model_identity import classify_model_config, model_reference
from ..models import (
    ASR_MODEL_REVISION,
    MODEL_ID,
    is_model_access_error,
    model_access_message,
)


class MemoizedEncoderProjection(torch.nn.Module):
    """Project each encoder output once across its autoregressive decode."""

    def __init__(self, projection: torch.nn.Module) -> None:
        super().__init__()
        self.projection = projection
        self._source: torch.Tensor | None = None
        self._projected: torch.Tensor | None = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self._source:
            self._source = source
            self._projected = self.projection(source)
        assert self._projected is not None
        return self._projected

    def clear(self) -> None:
        self._source = None
        self._projected = None


def clear_encoder_projection_cache(model) -> None:
    projection = model.model.decoder.proj
    if isinstance(projection, MemoizedEncoderProjection):
        projection.clear()


def prepare_encoder_attention_mask_once(_module, _inputs, output):
    """Convert the encoder padding mask once instead of once per decoder token."""
    mask = getattr(output, "attention_mask", None)
    if mask is None or mask.ndim != 2:
        return output
    # The stock decoder checks this on CUDA for every token. One check here also
    # preserves its mask-free SDPA path for batches without encoder padding.
    if bool(mask.all()):
        output.attention_mask = None
    else:
        output.attention_mask = mask.to(dtype=torch.bool)[:, None, None, :]
    return output


def _apply_hot_path_optimizations(
    model,
    *,
    reference: str,
    projection_cache: bool,
    encoder_attention_mask_cache: bool,
) -> None:
    if projection_cache:
        try:
            projection = model.model.decoder.proj
        except AttributeError as exc:
            raise RuntimeError(
                f"{reference} is incompatible with the encoder-projection cache"
            ) from exc
        model.model.decoder.proj = MemoizedEncoderProjection(projection)
    if encoder_attention_mask_cache:
        try:
            encoder = model.model.encoder
        except AttributeError as exc:
            raise RuntimeError(
                f"{reference} is incompatible with the encoder-mask cache"
            ) from exc
        encoder.register_forward_hook(prepare_encoder_attention_mask_once)


def load_asr_config_and_processor(model_id: str, revision: str | None):
    """Load and validate the native Transformers configuration and processor."""
    from transformers import (
        AutoConfig,
        AutoProcessor,
        CohereAsrConfig,
        CohereAsrProcessor,
    )

    source_kwargs: dict[str, object] = {"trust_remote_code": False}
    if revision is not None:
        source_kwargs["revision"] = revision
    reference = model_reference(model_id, revision)
    configuration = AutoConfig.from_pretrained(model_id, **source_kwargs)
    if not isinstance(configuration, CohereAsrConfig):
        raise ValueError(
            f"{reference} uses {configuration.model_type!r}; "
            "expected a native Transformers Cohere ASR checkpoint"
        )
    processor = AutoProcessor.from_pretrained(model_id, **source_kwargs)
    if not isinstance(processor, CohereAsrProcessor):
        raise ValueError(
            f"{reference} provides {type(processor).__name__}; "
            "expected CohereAsrProcessor"
        )
    return configuration, processor


def load_asr(
    device: str,
    dtype: torch.dtype,
    model_id: str = MODEL_ID,
    revision: str | None = ASR_MODEL_REVISION,
    model_format: str = "dense",
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
    projection_cache: bool = True,
    encoder_attention_mask_cache: bool = True,
):
    from transformers import CohereAsrForConditionalGeneration

    reference = model_reference(model_id, revision)
    if model_format not in {
        "dense",
        "bitsandbytes-int8",
        "bitsandbytes-int4",
    }:
        raise ValueError(f"Unsupported ASR model format: {model_format!r}")
    if model_format != "dense" and device != "cuda":
        raise RuntimeError(
            f"{model_format} checkpoints currently require a CUDA device"
        )
    try:
        configuration, processor = load_asr_config_and_processor(model_id, revision)
        configured_format, _ = classify_model_config(configuration.to_dict(), reference)
        if configured_format != model_format:
            raise RuntimeError(
                f"Resolved model format changed from {model_format} to "
                f"{configured_format} before weight loading"
            )
        model_kwargs: dict[str, object] = {
            "config": configuration,
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
        }
        if revision is not None:
            model_kwargs["revision"] = revision
        if model_format != "dense":
            model_kwargs["device_map"] = {"": f"cuda:{torch.cuda.current_device()}"}
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )
    except Exception as exc:
        if is_model_access_error(exc):
            raise SystemExit(model_access_message(exc, model_id=model_id)) from exc
        raise

    if adapter_id is not None:
        try:
            from peft import PeftModel

            adapter_kwargs: dict[str, object] = {
                "is_trainable": False,
                "low_cpu_mem_usage": True,
            }
            if adapter_revision is not None:
                adapter_kwargs["revision"] = adapter_revision
            adapter_model = PeftModel.from_pretrained(
                model, adapter_id, **adapter_kwargs
            )
            model = adapter_model.merge_and_unload(safe_merge=True)
        except Exception as exc:
            if is_model_access_error(exc):
                raise SystemExit(
                    model_access_message(exc, model_id=adapter_id)
                ) from exc
            raise RuntimeError(
                "Cannot load and safely merge adapter "
                f"{model_reference(adapter_id, adapter_revision)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    _apply_hot_path_optimizations(
        model,
        reference=reference,
        projection_cache=projection_cache,
        encoder_attention_mask_cache=encoder_attention_mask_cache,
    )
    if model_format == "dense":
        model.to(device)
    model.eval()
    return processor, model
