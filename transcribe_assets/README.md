# Transcription runtime assets

`silero_vad_v6.onnx` is the sequence-form Silero VAD v6 export distributed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It allows the CPU runtime to evaluate many 512-sample frames in one ONNX call while preserving Silero's recurrent state and timestamp rules.

- Source revision: `SYSTRAN/faster-whisper@ed9a06cd89a93e47838f564998a6c09b655d7f43`
- Asset source: `faster_whisper/assets/silero_vad_v6.onnx`
- Runtime reference: `faster_whisper/vad.py`
- SHA-256: `914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9`
- Upstream project: [Silero VAD](https://github.com/snakers4/silero-vad)
- License: MIT; see `LICENSE.silero-vad` for the model and `LICENSE.faster-whisper` for the repository that distributes this exact sequence export. Both notices apply to the bundled asset.

The local runtime fixes two boundary-only edge cases in the reference runner: an exactly divisible waveform does not receive an extra frame, and creating the first frame's zero context does not mutate the last audio frame. Input assembly is also bounded to 10,000 frames per call so multi-hour recordings do not create a whole-recording `(frames, 576)` intermediate array.
