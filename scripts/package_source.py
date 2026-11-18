"""Package only project source/configuration for transfer to a cloud runtime."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PathSnap_source_bundle.zip"
EXCLUDED = {".git", ".venv", "data", "artifacts", "results", "reports", "visualizations", "__pycache__"}

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        archive.write(path, relative.as_posix())
print(f"Source bundle written to {OUTPUT}")
