"""把分片的 woff2 解壓成 TTF，並為每個字重建立 cmap 索引（字 → 分片檔名）。"""
import json
import sys
from pathlib import Path

from fontTools.ttLib import TTFont


def build(out_root: Path, name: str, src: Path) -> None:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for i, f in enumerate(sorted(src.glob("*.woff2"))):
        font = TTFont(f)
        font.flavor = None
        fn = f"{i:03d}.ttf"
        font.save(out / fn)
        for cp in font.getBestCmap():
            index.setdefault(chr(cp), fn)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    print(f"{name}: {len(index)} chars")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for arg in sys.argv[2:]:
        name, src = arg.split("=", 1)
        build(root, name, Path(src))
