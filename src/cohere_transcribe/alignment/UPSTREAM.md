# Alignment utility provenance

This package retains three modified Python normalization and span helpers plus one punctuation-data file whose lineage includes both Fairseq MMS and ctc-forced-aligner.

Fairseq MMS source: <https://github.com/facebookresearch/fairseq/tree/728b947019fd186753197add48c39cbb24ea43e2/examples/mms>

Fairseq revision: `728b947019fd186753197add48c39cbb24ea43e2`

ctc-forced-aligner source: <https://github.com/MahmoudAshraf97/ctc-forced-aligner/tree/11855d1de76af2b490dd2e8e2db2661805ae90a0>

ctc-forced-aligner revision: `11855d1de76af2b490dd2e8e2db2661805ae90a0`

The Fairseq source identifies Facebook, Inc. and its affiliates as the copyright holder. The ctc-forced-aligner source identifies Mahmoud Ashraf as its author.

License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/); see [`LICENSE`](LICENSE).

`punctuations.lst` is an exact copy. `norm_config.py` and `text_utils.py` retain the upstream Arabic/English normalization and timestamp behavior while removing unsupported language modes, split modes, non-romanized paths, unused metadata, and confidence aggregation. `alignment_utils.py` retains the upstream `Segment`, `merge_repeats`, and `get_spans` behavior without the unused `Segment.length` property. The application uses TorchAudio for the alignment kernel and does not need the upstream C++ extension or model-loading helpers.
