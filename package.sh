#!/usr/bin/env bash
# Package QuadForge into an installable addon zip.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'EOF'
import re, zipfile
from pathlib import Path

src = Path("quadforge/__init__.py").read_text()
version = ".".join(re.search(r'"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)', src).groups())
out = Path(f"quadforge-{version}.zip")
out.unlink(missing_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path("quadforge").rglob("*.py")):
        if "__pycache__" not in p.parts:
            z.write(p)
print(f"Wrote {out}")
EOF
