"""Dependency and runtime compatibility checks for selected features."""

from __future__ import annotations

import importlib
import shutil

from ._version import DISTRIBUTION_NAME
from .models import (
    SILERO_VERSION,
    TRANSFORMERS_VERSION,
    TranscriptionConfig,
    package_version,
    release_pair,
)


def preflight_forced_align() -> None:
    """Verify that TorchAudio's maintained CPU CTC operation is callable."""
    try:
        import torch
        from torchaudio.functional import forced_align

        emissions = torch.log_softmax(
            torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float32), dim=-1
        )
        targets = torch.tensor([[1]], dtype=torch.int64)
        path, scores = forced_align(emissions, targets, blank=0)
        if path.shape != (1, 2) or scores.shape != (1, 2):
            raise RuntimeError(
                f"unexpected forced_align output shapes {path.shape}, {scores.shape}"
            )
    except Exception as exc:
        raise SystemExit(
            "TorchAudio forced alignment is unavailable or incompatible with this "
            f"PyTorch build ({type(exc).__name__}: {exc}). Install matching torch "
            "and torchaudio releases."
        ) from exc


def preflight_runtime(
    args: TranscriptionConfig, require_model_runtime: bool = True
) -> None:
    """Import only dependencies required by the selected execution path."""
    import torch

    required: list[tuple[str, str, str]] = []
    if require_model_runtime or args.alignment == "word":
        required.append(
            (
                "transformers",
                "Cohere ASR",
                f"transformers=={TRANSFORMERS_VERSION}",
            )
        )
    model_format = args.model_format or "dense"
    if require_model_runtime and model_format.startswith("bitsandbytes-"):
        if args.device != "cuda":
            raise SystemExit(
                f"{model_format} checkpoints currently require --device cuda"
            )
        required.extend(
            [
                (
                    "accelerate",
                    "saved bitsandbytes model placement",
                    f"{DISTRIBUTION_NAME}[quantized]",
                ),
                (
                    "bitsandbytes",
                    "saved bitsandbytes model inference",
                    f"{DISTRIBUTION_NAME}[quantized]",
                ),
            ]
        )
    if require_model_runtime and args.adapter is not None:
        required.append(
            (
                "peft",
                "LoRA adapter inference",
                f"{DISTRIBUTION_NAME}[adapters]",
            )
        )
    if require_model_runtime and args.vad == "silero":
        if args.vad_engine == "torch":
            required.append(
                (
                    "cohere_transcribe.vad.torch_silero",
                    "packed Torch Silero VAD",
                    "the installed cohere_transcribe.vad package",
                )
            )
        required.append(
            (
                "cohere_transcribe.vad.vectorized_silero",
                "Silero timestamp runtime",
                "the installed cohere_transcribe.vad package",
            )
        )
        if args.vad_engine == "onnx":
            required.append(
                (
                    "onnxruntime",
                    "ONNX Silero VAD",
                    f"{DISTRIBUTION_NAME}[onnx]",
                )
            )
    elif require_model_runtime and args.vad == "auditok":
        required.append(
            (
                "auditok.core",
                "Auditok VAD",
                f"{DISTRIBUTION_NAME}[auditok]",
            )
        )
    if args.alignment == "word":
        required.extend(
            [
                (
                    "torchaudio",
                    "word alignment",
                    f"{DISTRIBUTION_NAME}[word]",
                ),
                (
                    "uroman",
                    "word-alignment romanization",
                    f"{DISTRIBUTION_NAME}[word]",
                ),
                (
                    "cohere_transcribe.alignment.alignment_utils",
                    "word alignment span utilities",
                    f"{DISTRIBUTION_NAME}[word]",
                ),
                (
                    "cohere_transcribe.alignment.text_utils",
                    "word alignment text utilities",
                    f"{DISTRIBUTION_NAME}[word]",
                ),
            ]
        )
    audio_decode_required = require_model_runtime or args.alignment == "word"
    if audio_decode_required and args.audio_backend == "torchcodec":
        required.append(
            (
                "torchcodec",
                "TorchCodec audio decoding",
                DISTRIBUTION_NAME,
            )
        )
    elif audio_decode_required and args.audio_backend == "librosa":
        required.append(
            (
                "librosa",
                "Librosa audio decoding",
                DISTRIBUTION_NAME,
            )
        )

    for module_name, feature, package in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise SystemExit(
                f"Cannot initialize {feature}: import {module_name!r} failed ({exc}).\n"
                f"  Install a compatible build with: pip install {package}"
            ) from exc

    if (
        require_model_runtime
        and args.vad == "silero"
        and args.vad_engine in {"torch", "jit"}
    ):
        from .vad.runtime import SileroBackendUnavailable, packaged_silero_jit_path

        try:
            packaged_silero_jit_path()
        except SileroBackendUnavailable as exc:
            raise SystemExit(
                f"Silero {SILERO_VERSION} package data is unavailable ({exc}). "
                f"Reinstall {DISTRIBUTION_NAME}."
            ) from exc

    from packaging.version import Version

    if require_model_runtime or args.alignment == "word":
        transformers_version = package_version("transformers")
        if transformers_version is None or Version(transformers_version) != Version(
            TRANSFORMERS_VERSION
        ):
            raise SystemExit(
                "The optimized Cohere model paths are validated only with "
                f"transformers=={TRANSFORMERS_VERSION}; "
                f"found {transformers_version or 'unknown'}"
            )
    if require_model_runtime and model_format.startswith("bitsandbytes-"):
        bitsandbytes_version = package_version("bitsandbytes")
        accelerate_version = package_version("accelerate")
        if bitsandbytes_version is None or Version(bitsandbytes_version) < Version(
            "0.49.2"
        ):
            raise SystemExit(
                "Saved quantized checkpoints require bitsandbytes>=0.49.2; "
                f"found {bitsandbytes_version or 'unknown'}"
            )
        if accelerate_version is None or Version(accelerate_version) < Version(
            "1.13.0"
        ):
            raise SystemExit(
                "Saved quantized checkpoints require accelerate>=1.13.0; "
                f"found {accelerate_version or 'unknown'}"
            )
    if require_model_runtime and args.adapter is not None:
        peft_version = package_version("peft")
        if peft_version is None or Version(peft_version) < Version("0.19.1"):
            raise SystemExit(
                f"LoRA adapters require peft>=0.19.1; found {peft_version or 'unknown'}"
            )

    if args.alignment == "word":
        torch_pair = release_pair(torch.__version__)
        torchaudio_version = package_version("torchaudio")
        audio_pair = release_pair(torchaudio_version or "")
        if (
            torch_pair is not None
            and audio_pair is not None
            and torch_pair != audio_pair
        ):
            raise SystemExit(
                "PyTorch and TorchAudio must use matching major/minor releases for "
                f"forced alignment; found torch {torch.__version__} and "
                f"torchaudio {torchaudio_version}"
            )
        preflight_forced_align()
    if not audio_decode_required:
        return
    if args.audio_backend in {"auto", "torchcodec"}:
        from .audio.backends import resolve_audio_backend, torchcodec_is_usable

        if args.audio_backend == "auto":
            try:
                resolve_audio_backend("auto")
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        elif not torchcodec_is_usable():
            raise SystemExit(
                "--audio-backend torchcodec requires a working TorchCodec >= 0.14 "
                "installation and compatible system FFmpeg libraries"
            )
    elif args.audio_backend == "ffmpeg" and not shutil.which("ffmpeg"):
        raise SystemExit(
            "--audio-backend ffmpeg requires the ffmpeg executable on PATH"
        )
