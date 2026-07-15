# Usage Guide

This guide covers installation, transcription modes, batch processing, recovery, tuning, and common runtime problems. Run `cohere-transcribe --help` for every CLI option and its default value.

## Install

The supported release environment is Linux with Python 3.10 through 3.13. A CUDA GPU is strongly recommended. A CPU code path is implemented, but full-model CPU inference was not validated for this release and is unlikely to be practical for high-throughput work.

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cohere-transcribe
```

Automatic decoding prefers TorchCodec, which needs compatible FFmpeg shared libraries from the operating system. The OS package also provides `ffmpeg` and `ffprobe`, which the package uses for fallback decoding and inexpensive duration probes.

Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

The base package includes Librosa because the Cohere feature extractor uses it to construct mel filters. Librosa decoding is available through `--audio-backend librosa`, but automatic decoding does not select it.

### Optional Features

Install only the features you need:

```bash
# Auditok energy-based segmentation
python -m pip install "cohere-transcribe[auditok]"

# Sequence-based ONNX Silero fallback
python -m pip install "cohere-transcribe[onnx]"

# MMS CTC word alignment
python -m pip install "cohere-transcribe[word]"

# Saved bitsandbytes INT8/INT4 checkpoints on CUDA
python -m pip install "cohere-transcribe[quantized]"

# PEFT LoRA adapters for dense checkpoints
python -m pip install "cohere-transcribe[adapters]"
```

Extras can be combined:

```bash
python -m pip install "cohere-transcribe[adapters,auditok,onnx,word]"
```

### Device-Specific PyTorch

A normal pip installation uses the PyTorch wheel selected by the public package index. If that build does not match the host driver or accelerator, install the Torch 2.11 build for the target CUDA, ROCm, or CPU environment first, then install this package. Use the official PyTorch installation selector and keep Torch and TorchAudio on the same major/minor release.

For source development, uv can select a suitable Torch backend:

```bash
uv venv --python 3.12
uv pip install --editable ".[adapters,auditok,onnx,quantized,word]" --group dev --torch-backend=auto
```

The repository's `uv.lock` represents the standard PyPI source-development resolution and is checked by CI for consistency. CI installs its explicit CPU Torch backend from public dependency metadata. Use `uv run --no-sync` after a device-aware installation so uv does not replace the selected Torch build.

## Model Access

The default [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) repository is gated. When using that model, accept its terms, create a Hugging Face read token, and authenticate the same account. A public custom checkpoint does not require access to the default repository.

```bash
hf auth login
```

Set `HF_TOKEN` on non-interactive systems. Set `HF_HOME` when the model cache should use another disk. The first transcription downloads several gigabytes; word alignment downloads an additional model.

The package pins the exact default ASR and alignment model revisions used by the validated runtime. A selected custom Hub model or adapter branch, tag, or default branch is resolved to an immutable commit before inference. A local model or adapter is loaded directly from its canonical directory and does not use a revision.

Arabic is the default language. Use `--language en` for English audio; the package does not detect language automatically, and one language prompt applies to the complete command. It does not perform speaker diarization. Arabic-English code-switching is accepted as audio input, but output-language consistency is model-dependent and was not validated as a separate capability.

## Model Selection

The runtime supports three native Transformers loading paths:

| Loading path | Installation | Device | Loader behavior |
|---|---|---|---|
| Dense Cohere ASR checkpoint | Base package | CUDA, MPS, or CPU code path | Loads native `CohereAsrForConditionalGeneration` weights and applies the optimized Cohere inference path |
| Saved bitsandbytes INT8 or INT4 checkpoint | `cohere-transcribe[quantized]` | CUDA only | Detects the saved quantization configuration and lets Accelerate place the already-quantized weights |
| PEFT LoRA adapter over a dense checkpoint | `cohere-transcribe[adapters]` | Base model's supported device | Validates the adapter metadata, safely merges it into the dense base, then applies the optimized Cohere inference path |

The selected Hub repository or local directory must provide a native `cohere_asr` configuration, `CohereAsrProcessor`, and a Transformers safetensors or PyTorch weight entry point. Remote model code is never enabled. ONNX-only ASR, GGUF, and MLX checkpoints are rejected before audio preparation.

### Dense Checkpoints

Use `--model` for another native dense checkpoint. Supplying an immutable commit gives the clearest reproducibility contract, while a branch, tag, or omitted revision is accepted and resolved once before work begins:

```bash
cohere-transcribe recordings/ \
  --model owner/cohere-asr-model \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --output-dir transcripts/
```

The default model is the only dense checkpoint covered by the package's full release accuracy and performance baselines. Other compatible dense checkpoints may be fine-tunes with different vocabulary behavior, output length, accuracy, and throughput; compatibility is not a quality endorsement.

### Local Model Directories

Pass an existing model directory directly. Dense, saved bitsandbytes INT8/INT4, and fine-tuned Cohere ASR directories use the same format detection and loader paths as Hub snapshots:

```bash
cohere-transcribe recordings/ \
  --model /models/cohere-asr \
  --output-dir transcripts/
```

An existing directory always wins over an identically spelled Hub repository. Relative directories are accepted and canonicalized before state planning. Use an explicit path such as `./models/cohere-asr` when a not-yet-existing relative path could also be interpreted as a Hub ID. Existing files and explicit missing paths fail before model loading.

Local sources do not use Hub revisions. Combining a local model with `--model-revision`, or a local adapter with `--adapter-revision`, is a configuration error. In the Python API, a `pathlib.Path` model or adapter is always treated as local and must already be a directory; strings retain the CLI's Hub-or-local interpretation. Transformers and PEFT receive the directory directly, and local model resolution performs no Hub request. Output JSON, profile JSON, returned API provenance, checkpoint contracts, and reusable resource keys use the canonical path with a null revision.

The package deliberately does not hash or monitor local artifacts. Replacing configuration or weights inside the same directory does not invalidate an existing checkpoint and does not refresh a model retained by a live `Transcriber`. After changing files in place, close and recreate the `Transcriber`; that is sufficient when `publication=None`. When publication is enabled, also select a fresh output directory or remove the corresponding hidden ASR checkpoint and manifest before rerunning with CLI `--existing overwrite` or API `PublicationOptions(existing="overwrite")`. Moving to another directory naturally creates another identity.

### Saved Bitsandbytes Checkpoints

Saved bitsandbytes repositories and local directories carry their INT8 or INT4 configuration in `config.json`. Select the source normally; there is no `--int8`, `--int4`, or runtime conversion option:

```bash
python -m pip install "cohere-transcribe[quantized]"

cohere-transcribe recordings/ \
  --model NAMAA-Space/cohere-transcribe-arabic-07-2026-int8 \
  --device cuda \
  --output-dir transcripts/
```

`NAMAA-Space/cohere-transcribe-arabic-07-2026-int4` exercises the same path for its saved INT4 checkpoint. The runtime rejects unsupported quantizers, ambiguous INT4/INT8 metadata, and non-CUDA execution before loading model weights. Quantized checkpoints cannot currently be combined with a PEFT adapter.

The measured INT8 and INT4 checkpoints reduced memory but did not improve throughput on the RTX 3060. Use them when model memory or batch capacity is the constraint, then tune batch size for that checkpoint rather than carrying over the dense optimum. See [Alternate Model Checkpoints](performance.md#alternate-model-checkpoints) and [Selected Model Variants](benchmarks.md#selected-model-variants).

### PEFT LoRA Adapters

An adapter must declare `peft_type=LORA` and `task_type=SEQ_2_SEQ_LM`. When the selected base is a Hub repository, the adapter's declared base repository and optional base revision must match. A local base has no comparable Hub identity, so structural PEFT loading and safe merge establish compatibility. The runtime loads the adapter read-only and calls PEFT's safe merge before model transfer and Cohere-specific patching:

```bash
python -m pip install "cohere-transcribe[adapters]"

cohere-transcribe recordings/ \
  --model owner/cohere-asr-base \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --adapter owner/cohere-asr-lora \
  --adapter-revision 89abcdef0123456789abcdef0123456789abcdef \
  --output-dir transcripts/
```

Safe merging avoids retaining LoRA modules during generation and restores the dense execution shape, but the resulting text and generation time still depend on the adapter. An independent Moroccan evaluation of one public Darija adapter produced severe insertion loops and much worse WER than its base, so every adapter needs in-domain evaluation before use. That result applies to the tested adapter, not to LoRA as a method.

Hub and local sources can be mixed. For example, either the base or adapter can be a local directory:

```bash
cohere-transcribe recordings/ \
  --model /models/cohere-asr-base \
  --adapter owner/cohere-asr-lora \
  --adapter-revision 89abcdef0123456789abcdef0123456789abcdef
```

```bash
cohere-transcribe recordings/ \
  --model owner/cohere-asr-base \
  --adapter /models/cohere-asr-lora
```

## Recommended Commands

### Fast Approximate Subtitles

Silero VAD is the default. For continuous long-form recordings, the measured configuration below merges consecutive speech spans to provide more recognition context:

```bash
cohere-transcribe input.wav \
  --language ar \
  --vad silero \
  --vad-merge
```

It writes `input.txt`, `input.srt`, and `input.vtt`. Segment timestamps are approximate. Words are distributed only across retained speech spans; `--max-gap` controls whether a cue crosses a detected pause.

The default `--existing error` stops before replacing an output. Add `--existing overwrite` for an intentional rerun or `--existing skip` to accept only a complete manifest-verified generation.

### Plain Text

Keep Silero speech selection while skipping subtitle and alignment work:

```bash
cohere-transcribe input.wav --language ar --vad-merge --text-only
```

For clean continuous speech, fixed windows remove VAD work and can be faster:

```bash
cohere-transcribe input.wav --language ar --vad none --text-only
```

Fixed windows retain silence and may split words at a boundary, so use this mode only when its transcript-quality tradeoff is acceptable.

### Word-Level Timestamps

After installing the `word` extra, enable MMS CTC forced alignment:

```bash
cohere-transcribe input.wav \
  --language ar \
  --vad-merge \
  --alignment word \
  --align-dtype fp32
```

FP32 is the timestamp reference. `--align-dtype fp16` is faster on CUDA and closely matched FP32 on the validation corpus, but small timestamp shifts are possible. A segment shorter than two CTC frames or otherwise unalignable falls back to approximate timing; inspect warnings, profile fallback counts, and JSON `timing_source` values when exact provenance matters.

## Files and Directories

You can mix files and directories in one command:

```bash
cohere-transcribe interview.wav lecture.mp3 recordings/ --output-dir transcripts/
```

Directories are recursive by default. Use `--no-recursive` to process only files directly inside each supplied directory.

Directory discovery recognizes `.aac`, `.aif`, `.aiff`, `.alac`, `.flac`, `.m4a`, `.mp3`, `.mp4`, `.oga`, `.ogg`, `.opus`, `.wav`, `.wave`, `.webm`, and `.wma`. An explicitly supplied file is passed to the selected decoder even when its suffix is not on that discovery list.

Directory inputs preserve their relative subtree under `--output-dir`; an explicitly supplied file uses its basename. Without an output directory, each file writes beside its source. Canonical duplicate paths are processed once. Output collisions, input/output collisions, symlink targets, and nonregular output paths are rejected before model loading.

When separate directory roots contain the same relative file names, do not map both roots into one output directory; use separate output roots or commands so their output stems cannot collide.

If one file fails during a mixed batch, completed files remain published, the summary identifies failures, and the command exits nonzero.

## Python API

The public API exposes the same input discovery, validation, decoding, VAD, batching, generation, alignment, and output behavior as the CLI. It is synchronous and returns immutable result objects rather than printing a command summary.

### Inputs and Results

Pass one `str`, one `os.PathLike` such as `pathlib.Path`, or an ordered list or tuple of those values. Every value can be a file or directory, and directory expansion follows `TranscriptionOptions.recursive`, which defaults to `True`.

```python
from pathlib import Path

from cohere_transcribe import transcribe

one = transcribe("interview.wav").single
batch = transcribe([Path("lecture.mp3"), "recordings/"])
another_batch = transcribe(("first.wav", "second.wav"))
```

An empty string, bytes value, empty sequence, or non-path item is rejected before runtime initialization. Canonical duplicate paths are still processed once. `TranscriptionRun` is sequence-like, so it supports iteration, indexing, slicing, and `len()`. `run.single` returns the only result after directory expansion and duplicate removal; it raises `ValueError` when the run contains zero or multiple results.

Each `TranscriptionResult` has:

| Field | Meaning |
|---|---|
| `path`, `relative_path` | Canonical source path and its planned relative path |
| `status` | `completed`, `failed`, or `skipped` |
| `text` | Complete transcript when available |
| `duration` | Source duration known to the run |
| `segments` | ASR segment text and source intervals |
| `words` | Word or approximate word intervals for timestamped modes |
| `cues` | Rendered subtitle cues for timestamped modes |
| `outputs` | Published output paths, empty for an in-memory run |
| `error` | Per-file failure text, otherwise `None` |
| `provenance` | Resolved model/adapter identity, detected model format, decoder, VAD, generation-safety, alignment-fallback, checkpoint, and publication facts |

`TranscriptionRun.successful`, `.failed`, and `.skipped` filter the result tuple. `run.errors` contains run-level failures such as profile publication, and `run.ok` is true only when no file and no run-level operation failed. `run.statistics` contains stage timings, process serialization wait, generation counts, retry counts, peak PyTorch CUDA allocation and reservation measurements, and `real_time_factor_x`, which is successful source duration divided by elapsed time. `requested_options` preserves the API request, while `resolved_options` reports runtime-normalized device, precision, VAD policy, output mode, and formats. Per-file provenance records the decoder and VAD engine that actually completed.

### Configuration and Publication

`TranscriptionOptions` represents the complete transcription option surface. CLI names map directly to underscore-separated Python fields: for example, `--audio-backend` is `audio_backend`, `--vad-merge` is `vad_merge=True`, `--no-recursive` is `recursive=False`, and `--no-pipeline-preparation` is `pipeline_preparation=False`.

The fields are grouped as follows:

| Area | `TranscriptionOptions` fields |
|---|---|
| Model identity | `model`, `model_revision`, `adapter`, `adapter_revision` |
| Input and runtime | `language`, `text_only`, `recursive`, `device`, `dtype`, `audio_backend`, `audio_memory_gb`, `preprocess_workers`, `pipeline_preparation` |
| Segmentation | `vad`, `vad_engine`, `vad_batch_size`, `vad_block_frames`, `vad_threads`, `vad_merge`, `min_dur`, `max_dur`, `max_silence`, `energy_threshold`, `vad_threshold`, `min_silence_ms`, `speech_pad_ms` |
| ASR batching | `batch_size`, `batch_max_size`, `batch_audio_seconds`, `batch_vram_target`, `adaptive_batch`, `pin_memory` |
| Generation safety | `max_new_tokens`, `max_retry_tokens`, `truncation_policy`, `stop_repetition_loops` |
| Timing and cues | `alignment`, `align_batch_size`, `align_dtype`, `max_chars`, `max_cue_dur`, `max_gap` |
| Filesystem publication | `publication` |

The API defaults to `publication=None`. In this mode it returns transcript data, timings, cues, provenance, and statistics in memory without creating transcript directories, output files, checkpoints, manifests, locks, or a profile. Hugging Face and dependency caches remain governed by their own environment settings and may still be populated when models are downloaded.

Use `PublicationOptions` to opt into the CLI's durable output workflow:

```python
from pathlib import Path

from cohere_transcribe import PublicationOptions, TranscriptionOptions, transcribe

options = TranscriptionOptions(
    language="ar",
    vad="silero",
    vad_merge=True,
    alignment="segment",
    publication=PublicationOptions(
        formats=("txt", "srt", "vtt", "json"),
        output_dir=Path("transcripts"),
        existing="skip",
        profile_json=Path("transcripts/run.profile.json"),
    ),
)
run = transcribe(["interview.wav", "recordings/"], options=options)
```

The same fields select all three loading paths. For a saved quantized checkpoint, set only `model` and optionally `model_revision`; format detection is automatic. For an adapter, set its dense `model` plus `adapter` and optionally `adapter_revision`:

```python
from pathlib import Path

from cohere_transcribe import TranscriptionOptions, transcribe

quantized = transcribe(
    "recordings/",
    options=TranscriptionOptions(
        model="NAMAA-Space/cohere-transcribe-arabic-07-2026-int4",
        device="cuda",
        text_only=True,
    ),
)

adapted = transcribe(
    "recordings/",
    options=TranscriptionOptions(
        model="owner/cohere-asr-base",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        adapter="owner/cohere-asr-lora",
        adapter_revision="89abcdef0123456789abcdef0123456789abcdef",
    ),
)

local = transcribe(
    "recordings/",
    options=TranscriptionOptions(
        model=Path("/models/cohere-asr"),
        adapter=Path("/models/cohere-asr-lora"),
        device="cuda",
    ),
)
```

`requested_options` retains the references supplied by an API caller, including `Path` objects. `resolved_options` contains immutable Hub commits or canonical local paths with null revisions. Per-file provenance exposes the resolved identity and detected format; JSON outputs and profiles additionally retain the complete saved quantization configuration. Checkpoint and manifest contract keys bind the same configuration without duplicating it in their payloads.

`PublicationOptions.formats=None` uses the same mode-sensitive defaults as the CLI: TXT for text-only output and TXT/SRT/VTT for timestamped output. `output_dir=None` publishes beside each source. `existing` accepts `error`, `overwrite`, or `skip`; `profile_json` enables the separately atomic run profile.

A verified `existing="skip"` result has `status == "skipped"` and reports its planned output paths, but its transcript, segments, words, and cues are not read back into memory. Use the existing files or choose `overwrite` when the call must return content.

### Partial Failures and Exceptions

Per-file decode, segmentation, or inference failures do not discard successful siblings. By default, `transcribe()` returns the complete `TranscriptionRun`; inspect `run.ok`, `run.failed`, and each result's `error`.

```python
run = transcribe(["good.wav", "damaged.wav"])
for result in run.failed:
    print(result.path, result.error)
```

Set `raise_on_error=True` when an aggregate exception is more convenient. `BatchTranscriptionError.run` retains the same immutable partial run, including completed results:

```python
from cohere_transcribe import BatchTranscriptionError, transcribe

try:
    run = transcribe(["good.wav", "damaged.wav"], raise_on_error=True)
except BatchTranscriptionError as exc:
    for result in exc.run.successful:
        print("completed:", result.path)
    for result in exc.run.failed:
        print("failed:", result.path, result.error)
```

Semantic option validation raises `TranscriptionConfigurationError`, invalid input or publication planning raises `TranscriptionInputError`, and dependency, device, model-access, or runtime initialization failures raise `TranscriptionRuntimeError`. A progress callback that raises is reported as `ProgressCallbackError`. `PublicationOptions` rejects invalid formats or existing-output policies with `ValueError`; invalid option or callback object types raise `TypeError`. `KeyboardInterrupt` is propagated after runtime cleanup.

### Reusable Transcriber

The one-shot `transcribe()` helper always releases model resources before returning. Use `Transcriber` when one process will make repeated calls with the same options:

```python
from cohere_transcribe import Transcriber, TranscriptionOptions

options = TranscriptionOptions(language="ar", vad_merge=True)

with Transcriber(options) as transcriber:
    first = transcriber.transcribe("first.wav").single
    second_run = transcriber.transcribe(["second.wav", "third.wav"])
```

Construction is lightweight, and the ASR model loads on the first call that needs inference. A session configured for text-only or segment timing reuses the loaded model across compatible calls; its options do not change between calls. The package retains at most one ASR model per process, so switching sessions or running word alignment can require the next ASR call to reload it. `close()` releases retained resources and is idempotent; transcription after closing raises `TranscriberClosedError`.

### Progress and Concurrency

The API is quiet by default. Supply a callback to receive serialized `ProgressEvent` objects with either a message or bounded `current` and `total` values:

```python
from cohere_transcribe import ProgressEvent, Transcriber


def report(event: ProgressEvent) -> None:
    if event.message is not None:
        print(event.message)
    elif event.total is not None:
        print(f"{event.stage}: {event.current}/{event.total}")


with Transcriber(progress=report) as transcriber:
    run = transcriber.transcribe("recordings/")
```

Callbacks never overlap within a run and should return promptly so they do not delay inference. Transcription calls are serialized within one Python process; concurrent calls wait for the active call. A callback must not call `transcribe()` or `close()` on a transcriber participating in the active run, and recursive use is rejected with `TranscriberBusyError`.

On first runtime initialization, the package sets conservative PyTorch allocator, MPS fallback, and tokenizer parallelism environment defaults only when the application has not already supplied them. Set `PYTORCH_ALLOC_CONF` or `PYTORCH_CUDA_ALLOC_CONF`, `PYTORCH_ENABLE_MPS_FALLBACK`, and `TOKENIZERS_PARALLELISM` before the first call when an embedding application needs different process-wide policies.

Separate operating-system processes are not globally serialized and may compete for accelerator and host memory. When publication is enabled, the first process to claim an output stem proceeds and concurrent contenders sharing its registry namespace fail before model loading; different stems can proceed independently. Multi-host publication and containers with isolated `/tmp` namespaces require external coordination. Size process-level parallelism for the available hardware.

## Output Files

Timestamped modes write TXT, SRT, and VTT by default. Plain-text mode writes only TXT. Select formats explicitly when needed:

```bash
cohere-transcribe input.wav --formats txt srt vtt json
```

Without `--output-dir`, outputs are written beside each source file. With an output directory, directory inputs preserve their relative subtree and explicitly supplied files use their basename.

JSON includes transcript, segment, word or approximate timing, cue, generation-safety, segmentation, and runtime provenance. `--profile-json` writes exact stage timings, batch history, versions, decoder/VAD provenance, and memory telemetry for jobs processed by that command. Verified skipped jobs are removed before execution; if every job is skipped, the command returns without creating or updating a profile.

Output names use the source stem directly:

```text
input.txt
input.srt
input.vtt
input.json
```

Use `--max-chars`, `--max-cue-dur`, and `--max-gap` to control subtitle cue construction. These settings affect rendering, not ASR text.

## Existing Outputs and Resume

`--existing` controls what happens when requested outputs already exist:

| Policy | Behavior |
|---|---|
| `error` | Stop before model loading; this is the default |
| `overwrite` | Replace the requested output generation, reusing a compatible ASR checkpoint when possible |
| `skip` | Skip only a complete generation whose source, settings, manifest, formats, and hashes still match |

Each output stem uses hidden state files:

```text
.<stem>.cohere-transcribe.asr.json
.<stem>.cohere-transcribe.manifest.json
```

The ASR checkpoint allows subtitle settings, formats, and alignment to be retried without retranscribing when the source and ASR configuration are unchanged. Changes to language, resolved device/precision, decoder, VAD/segmentation, generation, batching, or transfer settings are examples that invalidate the ASR contract. The manifest verifies a complete published output set.

Publication locks use deterministic byte ranges in one private per-user registry at `/tmp/cohere-transcribe-<uid>/outputs.lock` on Linux. The kernel releases ranges after normal cleanup, exceptions, signals, or process termination. The registry file persists between runs and should not be deleted while transcription processes may be active. A single descriptor covers every output directory in a batch.

Delete checkpoint and manifest files only when no transcription process is running. Deleting a checkpoint removes render-only resume. Deleting a manifest makes `--existing skip` rebuild the output set.

The CLI exits with status 130 for SIGINT and 143 for SIGTERM after stopping queued work and active child processes. Completed files remain available, and the manifest written last prevents an interrupted output generation from being accepted by `--existing skip`. Use `--audio-backend ffmpeg` when decoding must be interruptible during a file; an in-process TorchCodec call can observe cancellation only before or after the native decode.

## Audio Decoding

The default `--audio-backend auto` policy is:

1. Use TorchCodec when it initializes successfully.
2. Select the OS `ffmpeg` executable when TorchCodec is unavailable or cannot initialize.
3. When TorchCodec was selected, retry an individual decode failure with FFmpeg when available.
4. Fail clearly when neither decoder is usable.

Explicit `torchcodec`, `ffmpeg`, and `librosa` modes are strict and never switch to another backend.

TorchCodec and FFmpeg enforce `--audio-memory-gb` while decoding. Librosa materializes audio before checking the limit, so it cannot provide the same memory bound when duration metadata is missing or inaccurate.

### Why TorchCodec with FFmpeg Recovery

TorchCodec was materially faster in the multi-file decoder benchmark. FFmpeg recovered one WAV that TorchCodec could not decode, so the combined policy provides fast normal decoding and a robust fallback without adding another Python decoder dependency. See [Performance](performance.md#audio-decoding) for the measured comparison.

## Segmentation and VAD

Silero is the default speech detector:

```bash
--vad silero --vad-engine auto
```

Automatic Silero selects the packed CPU PyTorch engine. If packed CPU PyTorch cannot initialize or is unavailable for an automatic request, the runtime can fall back to sequence-based ONNX when its optional dependency is installed, then to the packaged TorchScript engine. Explicit engines fail instead of falling through.

`--vad-merge` greedily joins consecutive speech spans whenever their combined start-to-end interval fits within `--max-dur`, retaining the intervening audio. This can reduce the ASR row count and provide more recognition context, but fewer rows did not always reduce wall time in the retained measurements. The raw speech spans remain available to approximate subtitle timing.

Auditok provides lightweight energy segmentation after installing its extra:

```bash
cohere-transcribe input.wav --vad auditok
```

`--vad none` retains all audio and creates contiguous fixed windows. It can reduce preparation overhead for clean or already clipped speech, but it can also split words at fixed boundaries and expose the model to silence, so it is not the general-purpose default.

### Why Packed CPU PyTorch Silero

Packed CPU PyTorch batches independent files while preserving one Silero state per recording. It performed best for the offline folder workload and avoids a required ONNX Runtime dependency. ONNX remains an optional explicit or fallback engine because its outputs matched the PyTorch timestamps on the validation corpus.

## Performance and Memory

The package automatically starts ASR batches at 24 rows on CUDA, 8 on MPS, and 4 on CPU. It also applies a padded-audio budget, splits on OOM, and remembers lower safe caps for later batches.

On the validated RTX 3060 12 GB, automatic CUDA batching already starts at 24. Pass the explicit value only to pin the measured configuration:

```bash
cohere-transcribe recordings/ --batch-size 24 --vad-merge
```

| Device | Guidance |
|---|---|
| NVIDIA RTX 3060 12 GB | BF16, static batch 24 is the measured configuration |
| Other CUDA GPUs | Not benchmarked by this project; start with automatic precision and lower `--batch-size` after OOM or when device headroom is small |
| CPU | FP32 code path; full 2B-model execution was not validated for this release |
| Apple MPS | FP16 path is implemented but was not run on physical MPS hardware for this release |
| AMD ROCm | Experimental through PyTorch's CUDA-compatible interface; not validated for this release |

Adaptive growth is available through `--adaptive-batch`, with an optional `--batch-max-size`, but it is not the validated default. Pinned host transfers are also opt-in through `--pin-memory` because the extra copy did not improve the reference GPU.

Multi-file preparation uses one worker for one file and at most two workers by default. `--pipeline-preparation` overlaps the next bounded decode/VAD group with GPU ASR. Lower `--batch-size` or `--align-batch-size` when GPU memory is constrained. Lower `--audio-memory-gb` only to reduce host PCM retention, and keep it above the decoded float32 size of every input file or that file will be rejected.

`--audio-memory-gb` is a hard decoded-PCM limit for each file and a scheduling target for prepared groups, not a hard cap on process RSS or aggregate decoder transients. Duration metadata guides grouping, so missing or inaccurate metadata can temporarily exceed the group target; Librosa also materializes a waveform before checking its size. Use fewer preparation workers, FFmpeg decoding, and operating-system memory limits when processing large inputs under a strict host-memory budget.

Packed CPU PyTorch VAD requires `--vad-batch-size * --vad-block-frames` to stay at or below 32,768 padded frames. The defaults are 16 files and 512 frames per file, and the runtime learns lower caps after CPU allocation failures.

The default `--dtype auto` selects BF16 on compatible CUDA GPUs, FP16 on other CUDA devices and Apple MPS, and FP32 on CPU. Use explicit precision only when reproducing a measured configuration or working around hardware support.

## Validate the Runtime

Check installed assets, dependencies, decoder availability, VAD, and accelerator detection without loading the 2B model:

```bash
cohere-transcribe-doctor
```

Also verify access to the pinned processor and configuration:

```bash
cohere-transcribe-doctor --model-access
```

Validate a custom model's native config and processor, immutable identity, format, and optional dependencies without loading its weights:

```bash
cohere-transcribe-doctor \
  --model NAMAA-Space/cohere-transcribe-arabic-07-2026-int8
```

Validate a model/adapter pair the same way:

```bash
cohere-transcribe-doctor \
  --model owner/cohere-asr-base \
  --adapter owner/cohere-asr-lora
```

Any model or adapter selection option implies `--model-access`; the explicit flag remains useful for checking the default model.

Validate word-alignment dependencies and pinned model access after installing its extra:

```bash
cohere-transcribe-doctor --mode word --model-access
```

Validate a specific decoder configuration:

```bash
cohere-transcribe-doctor --audio-backend ffmpeg
cohere-transcribe-doctor --audio-backend librosa
```

## Troubleshooting

### Model Access Is Denied

Accept the model terms with the same account used by `hf auth login`, then run `cohere-transcribe-doctor --model-access`.

### CUDA Is Not Available

Run `cohere-transcribe-doctor`. If it reports CPU-only execution when CUDA is expected, the PyTorch wheel or NVIDIA driver does not match the host. Recreate the environment using the Torch build recommended by the official PyTorch installation selector, then reinstall this package.

### Torch and TorchAudio Do Not Match

Word alignment requires TorchAudio 2.11 to match Torch 2.11. Install both from the same device-specific index, then run `cohere-transcribe-doctor --mode word`.

### TorchCodec Cannot Load FFmpeg

Install a supported FFmpeg package through the operating system. The Python package installs TorchCodec but does not bundle the native FFmpeg libraries it loads.

### BF16 Is Unsupported

Use FP16 on CUDA hardware without BF16 support:

```bash
cohere-transcribe input.wav --dtype fp16
```

### ONNX Runtime Prints a DRM Warning

Some headless Linux systems produce an ONNX Runtime device-discovery warning for `/sys/class/drm`. This does not mean CPU Silero inference failed. The initial command summary reports selected policy; profile and output JSON provenance report the engine and execution provider that actually completed.

### A Container Cannot Be Decoded

Install the OS FFmpeg package and keep `--audio-backend auto`, or select `--audio-backend ffmpeg` explicitly. Explicitly supplied files are accepted regardless of extension, so decoder support is the final authority.

### Out of Memory

For CUDA OOM, reduce `--batch-size` or `--align-batch-size`. For host-memory pressure, reduce `--preprocess-workers` or `--audio-memory-gb`, but keep the audio limit large enough for the biggest file's decoded float32 PCM. The runtime already splits failed ASR/alignment batches and preserves successful files, while smaller initial limits avoid retry work.
