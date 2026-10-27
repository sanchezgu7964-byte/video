#!/usr/bin/env bash
# 從 npm 取得 Noto Serif TC / Noto Sans TC（Google Fonts 的分片 woff2），
# 再由 build_fonts.py 解成 TTF 並建立「字 → 檔案」索引。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
curl -sSfLO https://registry.npmjs.org/noto-serif-tc/-/noto-serif-tc-31.0.0.tgz
curl -sSfLO https://registry.npmjs.org/noto-sans-tc/-/noto-sans-tc-36.0.0.tgz
mkdir serif sans
tar xzf noto-serif-tc-31.0.0.tgz -C serif
tar xzf noto-sans-tc-36.0.0.tgz -C sans
python3 "$ROOT/tools/build_fonts.py" "$ROOT/fonts" \
  serif_light="$TMP/serif/package/noto_serif_tc_light" \
  serif_regular="$TMP/serif/package/noto_serif_tc_regular" \
  sans_medium="$TMP/sans/package/noto_sans_tc_medium"
cp serif/package/LICENSE_OFL.txt "$ROOT/fonts/OFL.txt"
