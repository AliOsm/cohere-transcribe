from __future__ import annotations

from unittest import mock

import pytest

from cohere_transcribe.output.rendering import (
    build_cues,
    generate_json,
    generate_srt,
    generate_vtt,
)


def test_subtitle_renderers_emit_exact_srt_and_vtt_content() -> None:
    cues = [
        {"start": 0.0, "end": 1.234, "text": "مرحبا بالعالم"},
        {"start": 3661.005, "end": 3662.5, "text": "Second cue"},
    ]

    assert generate_srt(cues) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,234\n"
        "مرحبا بالعالم\n\n"
        "2\n"
        "01:01:01,005 --> 01:01:02,500\n"
        "Second cue\n\n"
    )
    assert generate_vtt(cues) == (
        "WEBVTT\n\n"
        "00:00.000 --> 00:01.234\n"
        "مرحبا بالعالم\n\n"
        "01:01:01.005 --> 01:01:02.500\n"
        "Second cue\n\n"
    )


def test_build_cues_splits_at_gaps_and_clamps_to_media_duration() -> None:
    words = [
        {"start": -0.1, "end": 0.1, "text": "first"},
        {"start": 0.2, "end": 0.4, "text": "cue."},
        {"start": 1.5, "end": 2.2, "text": "last"},
    ]

    assert build_cues(
        words,
        max_chars=100,
        max_duration=10.0,
        max_gap=0.5,
        media_duration=2.0,
    ) == [
        {"start": 0.0, "end": 0.4, "text": "first cue."},
        {"start": 1.5, "end": 2.0, "text": "last"},
    ]


def test_json_renderer_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        generate_json(mock.Mock(), (), (), (), payload={"duration": float("nan")})
