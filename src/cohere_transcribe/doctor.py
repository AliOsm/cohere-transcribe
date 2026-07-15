"""Validate an installed transcription runtime without loading the 2B model."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import shutil
import sys
from collections.abc import Sequence
from types import ModuleType

from packaging.version import Version

from ._version import DISTRIBUTION_NAME, __version__
from .audio.backends import (
    MIN_TORCHCODEC_VERSION,
    probe_torchcodec,
)
from .doctor_support import (
    ALIGN_VOCABULARY,
    EXPECTED_JIT_SHA256,
    EXPECTED_ONNX_SHA256,
    JIT_ASSET,
    ONNX_ASSET,
    Results,
)
from .model_identity import model_reference, resolve_model_identity
from .models import (
    ALIGN_MODEL_ID,
    ALIGN_MODEL_REVISION,
    ALIGN_PACKAGE_REPOSITORY,
    ALIGN_PACKAGE_REVISION,
    MODEL_ID,
    TRANSFORMERS_VERSION,
    UROMAN_VERSION,
    file_sha256,
    is_model_access_error,
    model_access_message,
    package_version,
    release_pair,
)


def import_required(results: Results, module: str, feature: str) -> ModuleType | None:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        results.fail(
            f"{feature}: cannot import {module!r}: {type(exc).__name__}: {exc}"
        )
        return None
    results.ok(f"{feature}: {module}")
    return imported


def validate_files(results: Results) -> None:
    installed_version = package_version(DISTRIBUTION_NAME)
    if installed_version is None:
        results.warn("package metadata is unavailable; running from a source checkout")
    elif installed_version == __version__:
        results.ok(f"package metadata version: {installed_version}")
    else:
        results.fail(
            f"package metadata is {installed_version}, runtime is {__version__}"
        )

    for name, path, expected in (
        ("TorchScript", JIT_ASSET, EXPECTED_JIT_SHA256),
        ("ONNX", ONNX_ASSET, EXPECTED_ONNX_SHA256),
    ):
        if not path.is_file():
            results.fail(f"missing Silero {name} asset: {path}")
            continue
        digest = file_sha256(path)
        if digest == expected:
            results.ok(f"Silero {name} asset integrity: {digest}")
        else:
            results.fail(
                f"Silero {name} checksum mismatch: expected {expected}, found {digest}"
            )


def validate_common_runtime(results: Results) -> ModuleType | None:
    results.ok(f"Python {sys.version.split()[0]}")

    torch = import_required(results, "torch", "PyTorch")
    import_required(results, "transformers", "Cohere ASR runtime")
    import_required(results, "sentencepiece", "model tokenizer")
    import_required(results, "google.protobuf", "processor serialization")
    import_required(results, "packaging", "version validation")
    import_required(results, "numpy", "numeric runtime")
    import_required(results, "tqdm", "progress display")

    version = package_version("transformers")
    if version is not None:
        from packaging.version import Version

        if Version(version) == Version(TRANSFORMERS_VERSION):
            results.ok(f"Transformers exact compatibility: {version}")
        else:
            results.fail(
                f"Transformers {version} does not match the validated "
                f"{TRANSFORMERS_VERSION} release"
            )

    if torch is None:
        return None
    if torch.cuda.is_available():
        results.ok(
            f"accelerator: CUDA device {torch.cuda.current_device()} - "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        results.ok("accelerator: Apple MPS")
    else:
        results.warn("accelerator: CPU only; the 2B model will be substantially slower")
    return torch


def validate_silero(results: Results, torch: ModuleType | None) -> None:
    onnx_available = False
    if importlib.util.find_spec("onnxruntime") is not None:
        onnx_available = (
            import_required(results, "onnxruntime", "optional ONNX Silero runtime")
            is not None
        )
    else:
        results.warn(
            "ONNX Runtime is not installed; packed Torch falls back directly to "
            "TorchScript if needed"
        )
    try:
        import numpy as np

        from .vad.torch_silero import BatchLimits, TorchSileroSequenceVAD
        from .vad.vectorized_silero import get_speech_timestamps_from_probabilities

        audios = [
            np.zeros(1024, dtype=np.float32),
            np.linspace(-0.02, 0.02, 1537, dtype=np.float32),
        ]
        if torch is None:
            raise RuntimeError("PyTorch is unavailable")
        before_threads = torch.get_num_threads()
        torch_model = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=16,
                max_files=2,
                max_valid_frames=32,
                max_padded_frames=32,
                max_audio_seconds=1.024,
            )
        )
        torch_probabilities = torch_model.speech_probabilities_batch(audios)
        after_threads = torch.get_num_threads()
        if before_threads != after_threads:
            raise RuntimeError(
                f"packed Torch loader changed thread count {before_threads} -> {after_threads}"
            )
        for audio, candidate in zip(audios, torch_probabilities, strict=True):
            expected_frames = math.ceil(len(audio) / 512)
            if candidate.shape != (expected_frames,) or not all(
                math.isfinite(float(value)) for value in candidate
            ):
                raise RuntimeError(
                    f"unexpected packed probability output {candidate!r}"
                )
            torch_timestamps = get_speech_timestamps_from_probabilities(
                len(audio), candidate
            )
            if not isinstance(torch_timestamps, list):
                raise RuntimeError("packed Torch timestamp smoke returned invalid data")

        if onnx_available:
            from .vad.vectorized_silero import VectorizedSileroVAD

            onnx_model = VectorizedSileroVAD()
            onnx_probabilities = [
                onnx_model.speech_probabilities(audio) for audio in audios
            ]
            for audio, candidate, reference in zip(
                audios, torch_probabilities, onnx_probabilities, strict=True
            ):
                if candidate.shape != reference.shape:
                    raise RuntimeError("packed Torch/ONNX output shape mismatch")
                difference = float(np.max(np.abs(candidate - reference), initial=0.0))
                if difference > 2e-6:
                    raise RuntimeError(
                        "packed Torch/ONNX probability difference "
                        f"{difference:.3g} exceeds 2e-6"
                    )
                onnx_timestamps = get_speech_timestamps_from_probabilities(
                    len(audio), reference
                )
                torch_timestamps = get_speech_timestamps_from_probabilities(
                    len(audio), candidate
                )
                if torch_timestamps != onnx_timestamps:
                    raise RuntimeError("packed Torch/ONNX timestamp smoke mismatch")
    except Exception as exc:
        results.fail(f"Silero runtime smoke test: {type(exc).__name__}: {exc}")
    else:
        if onnx_available:
            results.ok("packed Torch and bundled ONNX Silero agree and execute on CPU")
        else:
            results.ok("packed Torch Silero executes on CPU")


def validate_word_alignment(results: Results, torch: ModuleType | None) -> None:
    torchaudio = import_required(results, "torchaudio", "word alignment")
    alignment_utils = import_required(
        results,
        "cohere_transcribe.alignment.alignment_utils",
        "retained alignment span utilities",
    )
    text_utils = import_required(
        results,
        "cohere_transcribe.alignment.text_utils",
        "retained alignment text utilities",
    )
    import_required(results, "uroman", "Arabic alignment romanization")
    if package_version("uroman") != UROMAN_VERSION:
        results.fail(
            f"Uroman version: expected {UROMAN_VERSION}, "
            f"found {package_version('uroman') or 'missing'}"
        )
    else:
        results.ok(f"Uroman version: {UROMAN_VERSION}")
    if alignment_utils is not None and text_utils is not None:
        results.ok(
            f"alignment utilities: {ALIGN_PACKAGE_REPOSITORY}@{ALIGN_PACKAGE_REVISION}"
        )
        expected_exports = {
            alignment_utils: ("merge_repeats", "get_spans"),
            text_utils: ("preprocess_text", "postprocess_results"),
        }
        missing = [
            name
            for module, names in expected_exports.items()
            for name in names
            if not hasattr(module, name)
        ]
        if missing:
            results.fail(f"alignment utility exports are missing: {missing}")
        else:
            try:
                tokens, _ = text_utils.preprocess_text("مرحبا بكم في العالم", "ara")
                if tokens[-1] != "a l ' a l m":
                    raise RuntimeError(f"unexpected Uroman tokens: {tokens!r}")
            except Exception as exc:
                results.fail(
                    f"official Arabic romanization smoke test: {type(exc).__name__}: {exc}"
                )
            else:
                results.ok("retained Arabic Uroman path executes")
    if torch is None or torchaudio is None:
        return

    torch_pair = release_pair(torch.__version__)
    audio_pair = release_pair(torchaudio.__version__)
    if torch_pair != audio_pair:
        results.fail(
            f"torch {torch.__version__} and torchaudio {torchaudio.__version__} "
            "must have matching major/minor releases"
        )
        return
    results.ok(
        f"matched torch/torchaudio releases: {torch.__version__} / "
        f"{torchaudio.__version__}"
    )
    try:
        from torchaudio.functional import forced_align  # type: ignore[import-untyped]

        emissions = torch.log_softmax(
            torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float32), dim=-1
        )
        path, scores = forced_align(
            emissions, torch.tensor([[1]], dtype=torch.int64), blank=0
        )
        if path.shape != (1, 2) or scores.shape != (1, 2):
            raise RuntimeError(f"unexpected output shapes {path.shape}, {scores.shape}")
    except Exception as exc:
        results.fail(f"TorchAudio forced-align smoke test: {type(exc).__name__}: {exc}")
    else:
        results.ok("TorchAudio forced-align operation executes")


def report_optional_runtime(results: Results, audio_backend: str = "auto") -> None:
    if audio_backend not in {"auto", "torchcodec", "ffmpeg", "librosa"}:
        raise ValueError(f"unsupported audio backend: {audio_backend}")

    torchcodec_available = False
    if audio_backend in {"auto", "torchcodec"}:
        status = probe_torchcodec()
        torchcodec_available = status.usable
        torchcodec_version = status.version
        if torchcodec_available:
            results.ok(f"TorchCodec decoder: {torchcodec_version or 'installed'}")
        elif audio_backend == "torchcodec":
            results.fail(
                "TorchCodec decoder is unavailable or incompatible; install a working "
                f"TorchCodec >= {MIN_TORCHCODEC_VERSION} build ({status.detail})"
            )
        else:
            detail = (
                f"found {torchcodec_version}, but it is unusable"
                if torchcodec_version is not None
                else "not installed"
            )
            results.warn(
                f"TorchCodec is {detail} ({status.detail}); automatic decoding uses FFmpeg"
            )

    if audio_backend == "librosa":
        librosa = import_required(results, "librosa", "Librosa audio decoder")
        if librosa is not None:
            results.ok(f"Librosa version: {package_version('librosa') or 'unknown'}")
    elif (librosa_version := package_version("librosa")) is not None:
        results.ok(f"Librosa {librosa_version} (explicit decoder mode only)")

    if importlib.util.find_spec("auditok") is None:
        results.warn("Auditok is not installed; --vad auditok is unavailable")
    else:
        results.ok(f"optional Auditok {package_version('auditok') or 'installed'}")

    if audio_backend in {"auto", "ffmpeg"}:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            results.ok(f"FFmpeg executable: {ffmpeg}")
        elif audio_backend == "ffmpeg":
            results.fail("FFmpeg decoder is not on PATH")
        elif torchcodec_available:
            results.warn(
                "FFmpeg is not on PATH; automatic decoding has no per-file fallback"
            )
        else:
            results.fail(
                "No automatic audio decoder: install FFmpeg or a working TorchCodec"
            )


def validate_model_access(
    results: Results,
    include_aligner: bool,
    *,
    torch_runtime: ModuleType | None = None,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
) -> None:
    try:
        from transformers import AutoConfig, AutoTokenizer

        from .asr.model import load_asr_config_and_processor

        identity = resolve_model_identity(
            model_id,
            model_revision,
            adapter_id,
            adapter_revision,
        )
        _, processor = load_asr_config_and_processor(
            identity.model_id, identity.model_revision
        )
        maximum = getattr(processor.feature_extractor, "max_audio_clip_s", None)
        if maximum is None:
            raise RuntimeError("processor does not expose max_audio_clip_s")
        if include_aligner:
            aligner_config = AutoConfig.from_pretrained(
                ALIGN_MODEL_ID, revision=ALIGN_MODEL_REVISION
            )
            aligner_tokenizer = AutoTokenizer.from_pretrained(
                ALIGN_MODEL_ID,
                revision=ALIGN_MODEL_REVISION,
                word_delimiter_token=None,
            )
            expected_vocabulary = {
                token: index for index, token in enumerate(ALIGN_VOCABULARY)
            }
            if aligner_tokenizer.get_vocab() != expected_vocabulary:
                raise RuntimeError("pinned aligner tokenizer vocabulary changed")
            if aligner_tokenizer.pad_token_id != 1:
                raise RuntimeError("pinned aligner tokenizer pad ID changed")
            if getattr(aligner_config, "inputs_to_logits_ratio", None) != 320:
                raise RuntimeError("pinned aligner input stride changed")
    except Exception as exc:
        if is_model_access_error(exc):
            restricted_id = (
                adapter_id
                if adapter_id is not None and adapter_id in str(exc)
                else model_id
            )
            results.fail(model_access_message(exc, model_id=restricted_id))
        else:
            results.fail(f"selected model access: {type(exc).__name__}: {exc}")
    else:
        if identity.model_format.startswith("bitsandbytes-"):
            if torch_runtime is None or not torch_runtime.cuda.is_available():
                results.fail(
                    f"Selected {identity.model_format} model requires an available "
                    "CUDA device"
                )
            accelerate = import_required(
                results, "accelerate", "quantized model placement"
            )
            bitsandbytes = import_required(
                results, "bitsandbytes", "quantized model inference"
            )
            if accelerate is not None and bitsandbytes is not None:
                accelerate_version = package_version("accelerate")
                bitsandbytes_version = package_version("bitsandbytes")
                if (
                    accelerate_version is None
                    or Version(accelerate_version) < Version("1.13.0")
                    or bitsandbytes_version is None
                    or Version(bitsandbytes_version) < Version("0.49.2")
                ):
                    results.fail(
                        "Selected quantized model requires "
                        f"{DISTRIBUTION_NAME}[quantized] with "
                        "accelerate>=1.13.0 and bitsandbytes>=0.49.2; found "
                        f"accelerate {accelerate_version or 'missing'}, bitsandbytes "
                        f"{bitsandbytes_version or 'missing'}"
                    )
                else:
                    results.ok(
                        "quantized runtime: accelerate "
                        f"{accelerate_version}, bitsandbytes {bitsandbytes_version}"
                    )
        if identity.adapter_id is not None:
            peft = import_required(results, "peft", "adapter inference")
            if peft is not None:
                peft_version = package_version("peft")
                if peft_version is None or Version(peft_version) < Version("0.19.1"):
                    results.fail(
                        "Selected adapter requires "
                        f"{DISTRIBUTION_NAME}[adapters] with peft>=0.19.1; "
                        f"found {peft_version or 'missing'}"
                    )
                else:
                    results.ok(f"adapter runtime: peft {peft_version}")
        suffix = " and aligner" if include_aligner else ""
        adapter_suffix = (
            f", adapter {model_reference(identity.adapter_id, identity.adapter_revision)}"
            if identity.adapter_id is not None
            else ""
        )
        results.ok(
            f"ASR processor {model_reference(identity.model_id, identity.model_revision)} "
            f"({identity.model_format}){adapter_suffix}{suffix} accessible; "
            f"one-row limit is {maximum}s"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the transcription package without loading model weights."
    )
    parser.add_argument(
        "--mode",
        choices=("word", "segment"),
        default="segment",
        help="Validate dependencies for this output mode (default: segment).",
    )
    parser.add_argument(
        "--model-access",
        action="store_true",
        help="Also resolve and validate the selected Hugging Face processor/config files.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Hub repository or local model directory; implies --model-access.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hub model commit, tag, or branch; implies --model-access.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional Hub repository or local LoRA adapter directory.",
    )
    parser.add_argument(
        "--adapter-revision",
        default=None,
        help="Optional Hub adapter commit, tag, or branch; implies --model-access.",
    )
    parser.add_argument(
        "--audio-backend",
        choices=("auto", "torchcodec", "ffmpeg", "librosa"),
        default="auto",
        help="Validate this decoder configuration (default: auto).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = Results()
    validate_files(results)
    torch = validate_common_runtime(results)
    validate_silero(results, torch)
    if args.mode == "word":
        validate_word_alignment(results, torch)
    report_optional_runtime(results, args.audio_backend)
    selected_model_requested = (
        args.model != MODEL_ID
        or args.model_revision is not None
        or args.adapter is not None
        or args.adapter_revision is not None
    )
    if args.model_access or selected_model_requested:
        validate_model_access(
            results,
            include_aligner=args.mode == "word",
            torch_runtime=torch,
            model_id=args.model,
            model_revision=args.model_revision,
            adapter_id=args.adapter,
            adapter_revision=args.adapter_revision,
        )

    print()
    if results.failures:
        print(
            f"Validation failed: {results.failures} failure(s), "
            f"{results.warnings} warning(s)."
        )
        return 1
    print(f"Validation passed for {args.mode} mode with {results.warnings} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
