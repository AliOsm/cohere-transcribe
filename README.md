# Cohere Arabic/English batch transcription

This directory is the self-contained production runtime for `transcribe.py`. It transcribes files or directory trees with the pinned Cohere 2B ASR model and supports plain text, approximate segment timestamps, or MMS CTC word alignment.

The model weights are not redistributed here. The script downloads pinned revisions from Hugging Face on first use after you accept the Cohere model's access terms.

Word alignment uses the maintained `MahmoudAshraf97/ctc-forced-aligner` implementation pinned to an exact Git commit and its Uroman Arabic normalization path; `pip install -r requirements.txt` fetches and builds it automatically.

Silero `auto` uses bounded packed CPU Torch inference so independent recordings share encoder work while retaining separate recurrent state. Sequence ONNX and a side-effect-free TorchScript loader remain available as fallbacks or explicit comparison engines.

## Start here

1. Follow [`SETUP.md`](SETUP.md) to install the system packages, a device-appropriate PyTorch/TorchAudio pair, and `requirements.txt`.
2. Run `python validate_install.py --model-access`.
3. Start with the production command:

```bash
python transcribe.py input.wav \
  --language ar \
  --vad silero \
  --vad-merge \
  --alignment segment \
  --profile-json input.profile.json
```

For word-level timestamps, replace `--alignment segment` with `--alignment word`. For the fastest plain transcript, use `--text-only`; read the segmentation tradeoffs in `SETUP.md` before also disabling VAD.

## Bundle contents

| Path | Purpose |
|---|---|
| `transcribe.py` | Production CLI |
| `transcribe_assets/` | Packed Torch and sequence ONNX Silero runtimes, pinned ONNX model, and upstream licenses |
| `requirements.txt` | Cross-device Python runtime, excluding PyTorch/TorchAudio |
| `requirements-optional.txt` | Optional Auditok VAD mode |
| `validate_install.py` | Offline dependency and kernel smoke tests; optional model-access check |
| `VERSION.json` | Model revisions and validated environment |

The bundled Silero ONNX asset is MIT-licensed. It is a Silero model distributed by faster-whisper, so both directly applicable upstream notices and exact asset provenance are under `transcribe_assets/`. No license is asserted here for the Cohere or alignment model weights; their upstream terms apply when downloaded.

The batch default was selected from five reversed-order end-to-end runs over 500 files totaling 5,035.7 seconds. Packed Torch reduced median VAD compute from 7.15 to 2.26 seconds, reduced exposed preparation wait from 0.44 to 0.05 seconds, and preserved every VAD span and transcript byte; pipeline overlap limited the median total-wall improvement to about 0.2 seconds.
