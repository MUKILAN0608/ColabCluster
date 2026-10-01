"""Build the single Colab distribution without caches, secrets, or environments."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts" / "colabcluster-worker.zip"
    destination.parent.mkdir(exist_ok=True)
    files = [p for folder in ("common", "controller", "worker", "testing")
             for p in sorted((root / folder).glob("*.py"))]
    files += [root / name for name in ("requirements.txt", "pyproject.toml", "README.md", "LICENSE")]
    with ZipFile(destination, "w", ZIP_DEFLATED) as bundle:
        for source in files:
            bundle.write(source, source.relative_to(root).as_posix())
    with ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive verification failed")
    print(f"Created {destination}")


if __name__ == "__main__":
    main()
