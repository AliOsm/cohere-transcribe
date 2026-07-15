from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from cohere_transcribe.models import AudioJob, SourceSnapshot
from cohere_transcribe.output.publication import atomic_write_outputs
from cohere_transcribe.profiling import write_profile_json
from cohere_transcribe.state.io import create_state_temporary


def make_job(source: Path, output_paths: dict[str, Path]) -> AudioJob:
    return AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="segment",
        output_paths=output_paths,
        duration=1.0,
        segment_times=[(0.0, 1.0)],
        segment_texts=["new text"],
    )


def test_incomplete_rollback_reports_and_preserves_the_backup(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    txt_path = tmp_path / "clip.txt"
    srt_path = tmp_path / "clip.srt"
    txt_path.write_text("old text\n", encoding="utf-8")
    srt_path.write_text("old subtitles\n", encoding="utf-8")
    job = make_job(source, {"txt": txt_path, "srt": srt_path})
    real_replace = os.replace

    def fail_publication_and_restore(
        source_path: os.PathLike[str], destination_path: os.PathLike[str]
    ) -> None:
        source_candidate = Path(source_path)
        destination = Path(destination_path)
        if source_candidate.suffix == ".tmp" and destination == srt_path:
            raise OSError("simulated publication failure")
        if source_candidate.suffix == ".bak" and destination == txt_path:
            raise OSError("simulated rollback failure")
        real_replace(source_candidate, destination)

    with (
        mock.patch(
            "cohere_transcribe.output.publication.os.replace",
            side_effect=fail_publication_and_restore,
        ),
        pytest.raises(RuntimeError, match="rollback was incomplete") as caught,
    ):
        atomic_write_outputs(
            job,
            [{"start": 0.0, "end": 1.0, "text": "new text"}],
        )

    backups = list(tmp_path.glob(".*.bak"))
    assert "preserved backups" in str(caught.value)
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old text\n"
    assert txt_path.read_text(encoding="utf-8") == "new text\n"
    assert srt_path.read_text(encoding="utf-8") == "old subtitles\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert not job.published
    assert job.written == []


def test_interrupt_after_rename_restores_every_attempted_output(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    txt_path = tmp_path / "clip.txt"
    srt_path = tmp_path / "clip.srt"
    txt_path.write_text("old text\n", encoding="utf-8")
    srt_path.write_text("old subtitles\n", encoding="utf-8")
    job = make_job(source, {"txt": txt_path, "srt": srt_path})
    real_replace = os.replace

    def interrupt_after_second_rename(
        source_path: os.PathLike[str], destination_path: os.PathLike[str]
    ) -> None:
        source_candidate = Path(source_path)
        destination = Path(destination_path)
        real_replace(source_candidate, destination)
        if source_candidate.suffix == ".tmp" and destination == srt_path:
            raise KeyboardInterrupt

    with (
        mock.patch(
            "cohere_transcribe.output.publication.os.replace",
            side_effect=interrupt_after_second_rename,
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        atomic_write_outputs(
            job,
            [{"start": 0.0, "end": 1.0, "text": "new text"}],
        )

    assert txt_path.read_text(encoding="utf-8") == "old text\n"
    assert srt_path.read_text(encoding="utf-8") == "old subtitles\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []
    assert not job.published
    assert job.written == []


def test_missing_rollback_backup_is_reported_as_incomplete(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    txt_path = tmp_path / "clip.txt"
    srt_path = tmp_path / "clip.srt"
    txt_path.write_text("old text\n", encoding="utf-8")
    srt_path.write_text("old subtitles\n", encoding="utf-8")
    job = make_job(source, {"txt": txt_path, "srt": srt_path})
    real_replace = os.replace

    def remove_backup_before_failure(
        source_path: os.PathLike[str], destination_path: os.PathLike[str]
    ) -> None:
        source_candidate = Path(source_path)
        destination = Path(destination_path)
        if source_candidate.suffix == ".tmp" and destination == srt_path:
            txt_backup = next(tmp_path.glob(f".{txt_path.name}.*.bak"))
            txt_backup.unlink()
            raise OSError("simulated publication failure")
        real_replace(source_candidate, destination)

    with (
        mock.patch(
            "cohere_transcribe.output.publication.os.replace",
            side_effect=remove_backup_before_failure,
        ),
        pytest.raises(RuntimeError, match="rollback was incomplete"),
    ):
        atomic_write_outputs(
            job,
            [{"start": 0.0, "end": 1.0, "text": "new text"}],
        )

    assert txt_path.read_text(encoding="utf-8") == "new text\n"
    assert srt_path.read_text(encoding="utf-8") == "old subtitles\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []
    assert not job.published
    assert job.written == []


def test_failed_rename_does_not_replace_an_untouched_output_inode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    linked_path = tmp_path / "linked.txt"
    linked_path.write_text("old text\n", encoding="utf-8")
    output = tmp_path / "clip.txt"
    os.link(linked_path, output)
    original_inode = output.stat().st_ino
    job = make_job(source, {"txt": output})
    real_replace = os.replace

    def fail_before_rename(
        source_path: os.PathLike[str], destination_path: os.PathLike[str]
    ) -> None:
        source_candidate = Path(source_path)
        destination = Path(destination_path)
        if source_candidate.suffix == ".tmp" and destination == output:
            raise OSError("simulated pre-rename failure")
        real_replace(source_candidate, destination)

    with (
        mock.patch(
            "cohere_transcribe.output.publication.os.replace",
            side_effect=fail_before_rename,
        ),
        pytest.raises(OSError, match="pre-rename failure"),
    ):
        atomic_write_outputs(job, [])

    assert output.stat().st_ino == original_inode
    assert linked_path.stat().st_ino == original_inode
    assert output.read_text(encoding="utf-8") == "old text\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []
    assert not job.published
    assert job.written == []


def test_backup_close_failure_does_not_close_a_reused_descriptor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output = tmp_path / "clip.txt"
    output.write_text("old text\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("unrelated\n", encoding="utf-8")
    job = make_job(source, {"txt": output})
    real_close = os.close
    real_mkstemp = tempfile.mkstemp
    backup_descriptors: set[int] = set()
    reused_descriptors: list[int] = []

    def track_backup_descriptor(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = real_mkstemp(*args, **kwargs)
        if kwargs.get("suffix") == ".bak":
            backup_descriptors.add(descriptor)
        return descriptor, name

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor in backup_descriptors:
            backup_descriptors.remove(descriptor)
            reused = os.open(victim, os.O_RDONLY)
            if reused != descriptor:
                os.dup2(reused, descriptor)
                real_close(reused)
            reused_descriptors.append(descriptor)
            raise OSError("simulated backup close failure")

    with (
        mock.patch(
            "cohere_transcribe.output.publication.tempfile.mkstemp",
            side_effect=track_backup_descriptor,
        ),
        mock.patch(
            "cohere_transcribe.output.publication.os.close",
            side_effect=close_then_fail,
        ),
        pytest.raises(OSError, match="backup close failure"),
    ):
        atomic_write_outputs(job, [])

    assert output.read_text(encoding="utf-8") == "old text\n"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []
    assert backup_descriptors == set()
    assert len(reused_descriptors) == 1
    os.fstat(reused_descriptors[0])
    real_close(reused_descriptors[0])
    assert not job.published
    assert job.written == []


def test_output_mode_lookup_failure_creates_no_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = make_job(source, {"txt": tmp_path / "clip.txt"})

    with (
        mock.patch.object(Path, "stat", side_effect=PermissionError("mode denied")),
        mock.patch("cohere_transcribe.output.publication.tempfile.mkstemp") as mkstemp,
        pytest.raises(PermissionError, match="mode denied"),
    ):
        atomic_write_outputs(job, [])

    mkstemp.assert_not_called()


def test_state_mode_lookup_failure_creates_no_temporary_file(tmp_path: Path) -> None:
    marker = tmp_path / "state.json"

    with (
        mock.patch.object(Path, "stat", side_effect=PermissionError("mode denied")),
        mock.patch("cohere_transcribe.state.io.tempfile.mkstemp") as mkstemp,
        pytest.raises(PermissionError, match="mode denied"),
    ):
        create_state_temporary(marker, {"kind": "test"})

    mkstemp.assert_not_called()


def test_profile_mode_lookup_failure_creates_no_temporary_file(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"

    with (
        mock.patch.object(Path, "stat", side_effect=PermissionError("mode denied")),
        mock.patch("cohere_transcribe.profiling.tempfile.mkstemp") as mkstemp,
        pytest.raises(PermissionError, match="mode denied"),
    ):
        write_profile_json(profile, {"status": "ok"})

    mkstemp.assert_not_called()


def test_cleanup_failure_does_not_mask_publication_error(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = make_job(source, {"txt": tmp_path / "clip.txt"})
    source.write_bytes(b"changed after planning")
    real_unlink = Path.unlink

    def fail_temporary_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".tmp":
            raise PermissionError("simulated cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    with (
        mock.patch.object(Path, "unlink", fail_temporary_cleanup),
        pytest.raises(RuntimeError, match="Source changed"),
    ):
        atomic_write_outputs(job, [])

    for temporary in tmp_path.glob(".*.tmp"):
        temporary.unlink()


def test_cleanup_failure_does_not_mask_profile_error(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    real_unlink = Path.unlink

    def fail_temporary_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".tmp":
            raise PermissionError("simulated cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    with (
        mock.patch(
            "cohere_transcribe.profiling.json.dump",
            side_effect=RuntimeError("simulated serialization failure"),
        ),
        mock.patch.object(Path, "unlink", fail_temporary_cleanup),
        pytest.raises(RuntimeError, match="serialization failure"),
    ):
        write_profile_json(profile, {"status": "ok"})

    for temporary in tmp_path.glob(".*.tmp"):
        temporary.unlink()
