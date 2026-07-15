# cohere-transcribe

`cohere-transcribe` is an unofficial Python package for high-throughput offline transcription with Cohere's 2B Arabic/English ASR model. Its CLI and Python API process individual files, multiple paths, and nested directories with bounded-memory batching. Results can be returned or published as plain text, approximate segment-timed subtitles, or optional word-timed subtitles.

The default CohereLabs ASR weights are downloaded from a pinned Hugging Face revision after you accept that repository's model terms. Compatible model and adapter directories can also be loaded directly from local storage without a Hub lookup. The package includes the validated Silero VAD weights but does not redistribute ASR model weights.

## Requirements

- Linux with Python 3.10 through 3.13.
- Access to [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) when using the default model; a public custom checkpoint does not require access to the default repository.
- System FFmpeg libraries for TorchCodec. Installing the `ffmpeg` OS package also provides the command-line fallback used when TorchCodec is unavailable or rejects a file.
- A CUDA GPU is strongly recommended for the 2B model. A CPU code path exists, but full-model CPU inference was not validated for this release.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Install

Create a virtual environment and install the package from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cohere-transcribe
```

On a GPU host, the PyTorch wheel selected by the public package index must match the installed driver and accelerator. If it does not, install the appropriate Torch 2.11 build first, then install this package; see [Device-Specific PyTorch](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/usage.md#device-specific-pytorch).

The base installation includes TorchCodec, Librosa, and the dependencies required for the default segment-timestamp pipeline. Optional extras add Auditok segmentation, ONNX Runtime, word alignment, saved bitsandbytes checkpoints, or PEFT adapters:

```bash
python -m pip install "cohere-transcribe[auditok]"
python -m pip install "cohere-transcribe[onnx]"
python -m pip install "cohere-transcribe[word]"
python -m pip install "cohere-transcribe[quantized]"
python -m pip install "cohere-transcribe[adapters]"
```

The `quantized` extra enables saved bitsandbytes INT8/INT4 checkpoints on CUDA. The `adapters` extra enables PEFT LoRA loading and safe merging into a dense base model. Extras can be combined, for example `cohere-transcribe[adapters,auditok,onnx,word]`.

## Model Access

Accept the model terms on Hugging Face, create a read token, and authenticate the same account:

```bash
hf auth login
```

For non-interactive systems, set `HF_TOKEN`. Use `HF_HOME` when the model cache should live on a larger disk.

Arabic is the default language. Use `--language en` for English audio.

## Model Selection

The default is the evaluated `CohereLabs/cohere-transcribe-arabic-07-2026` commit. `--model` also accepts another Hugging Face repository or an existing local directory when it provides native Transformers `CohereAsrForConditionalGeneration` weights, configuration, and a `CohereAsrProcessor` without remote code:

```bash
cohere-transcribe input.wav \
  --model owner/cohere-asr-model \
  --model-revision 0123456789abcdef0123456789abcdef01234567
```

The package resolves a Hub branch, tag, or omitted custom revision to an immutable commit before inference. A local directory is canonicalized and passed directly to Transformers; `--model-revision` is invalid for a local model. Outputs and profiles record its canonical path with a null revision, and checkpoint and reusable-model contracts bind the same identity. No local files are hashed or copied. If local weights are replaced in place, close and recreate any live `Transcriber`; that is sufficient for in-memory use. When publication is enabled, also use a fresh output directory or remove the matching hidden ASR checkpoint and manifest before rerunning with CLI `--existing overwrite` or API `PublicationOptions(existing="overwrite")`. Custom dense models use the normal installation and execution path, but their recognition quality and runtime are properties of their weights and must be evaluated independently.

```bash
cohere-transcribe input.wav --model /models/cohere-asr
```

Saved bitsandbytes INT8 and INT4 checkpoints are detected from `config.json`; there is no runtime quantization flag. Install the extra and select the saved repository:

```bash
python -m pip install "cohere-transcribe[quantized]"

cohere-transcribe input.wav \
  --model NAMAA-Space/cohere-transcribe-arabic-07-2026-int8 \
  --device cuda
```

These checkpoints currently require CUDA. On the validated RTX 3060 they reduced memory but were slower than dense BF16, so they are memory-capacity options rather than speed presets.

LoRA adapters accept a Hub repository or local directory and require a dense base. The package validates a `SEQ_2_SEQ_LM` LoRA adapter, checks Hub base identity where it is meaningful, loads it read-only, safely merges it into the base model for offline throughput, and only then applies the Cohere hot-path optimizations:

```bash
python -m pip install "cohere-transcribe[adapters]"

cohere-transcribe input.wav \
  --model owner/cohere-asr-base \
  --adapter owner/cohere-asr-lora \
  --adapter-revision 0123456789abcdef0123456789abcdef01234567
```

Adapter compatibility does not establish adapter quality. Validate every fine-tune on independent in-domain references before deployment. ONNX ASR, GGUF, and MLX checkpoints are not supported by this runtime.

A local adapter uses no adapter revision:

```bash
cohere-transcribe input.wav \
  --model /models/cohere-asr \
  --adapter /models/cohere-asr-lora
```

## Quick Start

The default path uses Silero speech boundaries and creates approximate segment-timed subtitles. For continuous long-form recordings, the measured configuration below also combines consecutive spans when their complete interval fits the duration limit:

```bash
cohere-transcribe input.wav \
  --language ar \
  --vad-merge
```

This writes `input.txt`, `input.srt`, and `input.vtt`. Add `--formats txt srt vtt json` for provenance-rich JSON.

The default `--existing error` protects existing outputs. Use `--existing overwrite` to replace them or `--existing skip` to reuse only a complete manifest-verified generation.

Plain text with Silero speech selection:

```bash
cohere-transcribe input.wav --language ar --vad-merge --text-only
```

After installing the `word` extra, request word-level timestamps with:

```bash
cohere-transcribe input.wav --language ar --vad-merge --alignment word
```

## Batch Transcription

Pass any combination of files and directories. Directory traversal is recursive by default, and the model is loaded at most once when inference is needed:

```bash
cohere-transcribe a.wav b.mp3 recordings/ \
  --language ar \
  --vad-merge \
  --output-dir transcripts/ \
  --existing skip
```

Directory inputs preserve their relative subtree under the output directory; explicitly supplied files use their basename. Audio decoding, VAD, preparation, and ASR batching operate across files while each recording keeps independent segmentation state and output files. Successful files are published even when another file fails, and the command exits nonzero when any file fails.

## Python API

`transcribe()` accepts one string or path-like object, or an ordered list or tuple containing files and directories. It returns results in memory and creates no transcript files by default:

```python
from pathlib import Path

from cohere_transcribe import TranscriptionOptions, transcribe

run = transcribe(
    [Path("interview.wav"), "recordings/"],
    options=TranscriptionOptions(language="ar", vad_merge=True),
)

for result in run:
    print(result.path, result.status)
    if result.text is not None:
        print(result.text)
```

For one expanded audio file, `run.single` returns its `TranscriptionResult`. To write durable outputs, checkpoints, manifests, and an optional profile, add `PublicationOptions`:

```python
from cohere_transcribe import PublicationOptions, TranscriptionOptions, transcribe

options = TranscriptionOptions(
    model="owner/cohere-asr-model",
    model_revision="0123456789abcdef0123456789abcdef01234567",
    language="ar",
    vad_merge=True,
    publication=PublicationOptions(
        formats=("txt", "srt", "vtt", "json"),
        output_dir="transcripts/",
        existing="skip",
        profile_json="transcripts/run.profile.json",
    ),
)
run = transcribe("recordings/", options=options)
```

The same API fields select quantized models or adapters. Strings can identify either Hub repositories or existing local directories; a `pathlib.Path` always means an existing local directory. A saved quantized repository or directory needs only `model=...`; an adapter call additionally sets `adapter=...` and optionally uses `adapter_revision=...` for a Hub adapter. Each result's `provenance` reports the resolved Hub identity or canonical local path, optional revision, detected format, and merged adapter identity.

Use `Transcriber` as a context manager for repeated calls with one immutable option set. It loads models lazily and retains a compatible ASR model when the session is configured for text-only or segment timing:

```python
from cohere_transcribe import Transcriber, TranscriptionOptions

with Transcriber(TranscriptionOptions(vad_merge=True)) as transcriber:
    first = transcriber.transcribe("first.wav").single
    second = transcriber.transcribe(["second.wav", "third.wav"])
```

See the [Python API guide](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/usage.md#python-api) for result fields, partial failures, progress callbacks, resource lifetime, and concurrency behavior.

## Output Modes

| Mode | Command | Output |
|---|---|---|
| Segment timestamps | Default or `--alignment segment` | TXT, SRT, and VTT with fast approximate timing |
| Plain text | `--text-only` | TXT only, without alignment work |
| Word timestamps | `--alignment word` | TXT, SRT, and VTT using MMS CTC forced alignment |

Segment timing is the default because it keeps the fast ASR path and uses retained detected speech spans for approximate cue timing. Word alignment provides per-word CTC boundaries but loads another model and takes additional time; segments that cannot be aligned use an explicit approximate fallback. Fixed-window text mode with `--vad none --text-only` is faster on the measured clean continuous speech, but it can split words at window boundaries and transcribe silence.

## Validate the Installation

The doctor checks package data, dependency compatibility, the selected decoder, VAD, and accelerator availability without loading the 2B model:

```bash
cohere-transcribe-doctor
cohere-transcribe-doctor --model-access
```

Validate a custom Hub or local model or adapter without loading its weights:

```bash
cohere-transcribe-doctor \
  --model owner/cohere-asr-base \
  --adapter owner/cohere-asr-lora
```

Passing `--model`, `--model-revision`, `--adapter`, or `--adapter-revision` implies `--model-access`.

For the complete word-alignment dependency and model-access check:

```bash
cohere-transcribe-doctor --mode word --model-access
```

## Performance

On the validated RTX 3060 12 GB system, the installed package transcribed a 69-minute Arabic grammar lecture in a 32.27-second external median with approximate segment timing. A 500-file, 83.9-minute batch completed in a 39.27-second external median. Measured transcripts and subtitle files were byte-identical to their stored validation baselines; this is an implementation-stability check, not a human-reference WER claim.

In a separate 500-clip component harness that excluded model loading, VAD, alignment, and publication, dense BF16 completed in 32.46 seconds at 9.43 GiB peak CUDA allocation. Saved INT8 completed in 70.67 seconds at 7.76 GiB and saved INT4 completed in 43.66 seconds at 6.95 GiB with the same batch size. Neither quantized WER difference was statistically distinguishable from the dense result on that probe, but their hypotheses were not identical. One independently tested public Darija LoRA performed substantially worse than its dense base; it is documented as a failed adapter evaluation, not a general statement about LoRA.

See [Performance](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/performance.md) for configurations, methodology, resource measurements, and the reasons behind the default runtime choices. See [Accuracy Benchmarks](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/benchmarks.md) for the human-reference WER/CER evaluation and quality safeguards.

## Documentation

- [Usage guide](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/usage.md): CLI and Python API usage, modes, batching, recovery, tuning, and troubleshooting.
- [Architecture](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/architecture.md): runtime stages, module ownership, packaged assets, and design decisions.
- [Upstream work](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/upstream.md): ecosystem issues and pull requests, their current status, and their relationship to the local runtime.
- [Performance](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/performance.md): installed-wheel baselines, configuration studies, alternate engines, and reproducible timing guidance.
- [Accuracy benchmarks](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/benchmarks.md): datasets, normalization, WER/CER, confidence intervals, and official-result comparisons.
- [Development](https://github.com/AliOsm/cohere-transcribe/blob/main/docs/development.md): uv environment, tests, package builds, and releases.
- [Release reports](https://github.com/AliOsm/cohere-transcribe/tree/main/reports): versioned release-validation evidence.
- [Changelog](https://github.com/AliOsm/cohere-transcribe/blob/main/CHANGELOG.md): release-level user and developer changes.

## License

Original project code and documentation are licensed under Apache License 2.0. This distribution also contains retained word-alignment helpers under Creative Commons Attribution-NonCommercial 4.0 International and bundled Silero/faster-whisper VAD components under MIT licenses. CC BY-NC 4.0 includes a noncommercial restriction. Runtime-downloaded model weights retain the terms published by their owners; the default MMS alignment model is also CC BY-NC 4.0.

See [LICENSE](https://github.com/AliOsm/cohere-transcribe/blob/main/LICENSE), [NOTICE](https://github.com/AliOsm/cohere-transcribe/blob/main/NOTICE), and [THIRD_PARTY_NOTICES.md](https://github.com/AliOsm/cohere-transcribe/blob/main/THIRD_PARTY_NOTICES.md) for the exact scopes, source revisions, modifications, and included license texts.

Run `cohere-transcribe --help` for the complete CLI reference.
