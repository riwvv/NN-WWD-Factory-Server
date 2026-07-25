import os

from app.utils.file_utils import ensure_directory_exists, get_file_size, clean_directory


def test_ensure_directory_exists_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    ensure_directory_exists(str(target))
    assert target.is_dir()


def test_get_file_size_matches_written_bytes(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"0123456789")
    assert get_file_size(str(file_path)) == 10


def test_clean_directory_removes_contents_but_keeps_folder(tmp_path):
    (tmp_path / "file.txt").write_text("data")
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "nested.txt").write_text("data")

    clean_directory(str(tmp_path))

    assert tmp_path.is_dir()
    assert os.listdir(str(tmp_path)) == []


def test_clean_directory_on_missing_path_is_noop(tmp_path):
    missing = tmp_path / "missing"
    clean_directory(str(missing))
    assert not missing.exists()
