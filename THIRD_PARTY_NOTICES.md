# Third-Party Notices

This distribution contains components under more than one license. The Apache License 2.0 in [`LICENSE`](LICENSE) applies to original cohere-transcribe code and documentation. It does not replace the licenses identified below.

## Fairseq MMS and ctc-forced-aligner

The following files retain normalization, token-span, and punctuation behavior derived from Fairseq MMS and the maintained ctc-forced-aligner project:

- `src/cohere_transcribe/alignment/alignment_utils.py`
- `src/cohere_transcribe/alignment/norm_config.py`
- `src/cohere_transcribe/alignment/text_utils.py`
- `src/cohere_transcribe/alignment/punctuations.lst`

Sources:

- Fairseq MMS, revision `728b947019fd186753197add48c39cbb24ea43e2`: <https://github.com/facebookresearch/fairseq/tree/728b947019fd186753197add48c39cbb24ea43e2/examples/mms>
- ctc-forced-aligner, revision `11855d1de76af2b490dd2e8e2db2661805ae90a0`: <https://github.com/MahmoudAshraf97/ctc-forced-aligner/tree/11855d1de76af2b490dd2e8e2db2661805ae90a0>

The Fairseq source identifies Facebook, Inc. and its affiliates as the copyright holder. The ctc-forced-aligner source identifies Mahmoud Ashraf as its author. These files are distributed under Creative Commons Attribution-NonCommercial 4.0 International. See [`src/cohere_transcribe/alignment/LICENSE`](src/cohere_transcribe/alignment/LICENSE) and [`src/cohere_transcribe/alignment/UPSTREAM.md`](src/cohere_transcribe/alignment/UPSTREAM.md). The local copies remove unused language modes, metadata, confidence aggregation, and native-extension integration while retaining the behavior required by this package.

## Silero VAD

The packed Torch VAD implementation, shared timestamp behavior, and `src/cohere_transcribe/vad/silero_vad.jit` use the public Silero VAD 6.2.1 architecture, code, and model asset.

- Source revision: `v6.2.1` (`7e30209a3e901f9842f81b225f3e93d8199902b1`)
- Source: <https://github.com/snakers4/silero-vad/tree/v6.2.1>
- Copyright: 2020-present Silero Team
- License: MIT; see [`src/cohere_transcribe/vad/LICENSE.silero-vad`](src/cohere_transcribe/vad/LICENSE.silero-vad)

The local implementation adds bounded offline batching, independent state for concurrent audio files, cancellation, validation, deterministic ordering, and runtime telemetry. Asset provenance and hashes are recorded in [`src/cohere_transcribe/vad/README.md`](src/cohere_transcribe/vad/README.md).

## faster-whisper

`src/cohere_transcribe/vad/silero_vad_v6.onnx` is the Silero VAD v6 sequence export distributed by faster-whisper. The local ONNX runtime follows its public VAD integration while adding bounded input assembly and boundary fixes.

- Source revision: `ed9a06cd89a93e47838f564998a6c09b655d7f43`
- Source: <https://github.com/SYSTRAN/faster-whisper/tree/ed9a06cd89a93e47838f564998a6c09b655d7f43>
- Copyright: 2023 SYSTRAN
- License: MIT; see [`src/cohere_transcribe/vad/LICENSE.faster-whisper`](src/cohere_transcribe/vad/LICENSE.faster-whisper)

## Runtime Downloads

Model weights downloaded at runtime are not included in the Python distribution and retain their own terms. The default Cohere ASR model is licensed under Apache License 2.0. The default MMS alignment model is licensed under Creative Commons Attribution-NonCommercial 4.0 International. Custom models and adapters remain subject to the terms published by their owners.

Dependencies installed separately by a Python package manager are not copied into this distribution and retain their respective licenses.
