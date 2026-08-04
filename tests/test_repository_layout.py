from pathlib import Path


def test_source_tree_is_repository_owned() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / 'README.md').is_file()
    assert any((root / 'src').iterdir())
