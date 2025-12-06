#!/usr/bin/env bash
# usage: ./zip_rirs.sh                # 현재 경로의 ./opt 대상
#    or: ./zip_rirs.sh /path/to/opt  # 대상 경로 지정

set -Eeuo pipefail

ROOT_DIR="${1:-./opt}"      # 기본 대상 디렉터리
PATTERN='*rirs*'            # 파일명 패턴

# zip 유틸리티 확인
if ! command -v zip >/dev/null 2>&1; then
  echo "ERROR: 'zip' 명령을 찾을 수 없습니다. (Ubuntu: sudo apt-get install zip)"
  exit 1
fi

# 대상 디렉터리 존재 확인
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: 대상 디렉터리가 없습니다: $ROOT_DIR"
  exit 1
fi

# 찾고 압축
found_any=false
find "$ROOT_DIR" -type f -name "$PATTERN" -print0 | while IFS= read -r -d '' file; do
  found_any=true
  dir="$(dirname "$file")"
  base="$(basename "$file")"
  out="$dir/${base}.zip"

  if [[ -e "$out" ]]; then
    echo "Skip (이미 존재): $out"
    continue
  fi

  echo "Zipping: $file -> $out"
  # zip 내부에 경로를 넣지 않도록 해당 디렉터리로 이동 후 파일만 추가(-q: 조용히)
  ( cd "$dir" && zip -q -- "$out" "$base" )
done

# 파일이 하나도 없을 때 안내
if [[ "$found_any" = false ]]; then
  echo "패턴 '$PATTERN'에 맞는 파일을 찾지 못했습니다: $ROOT_DIR"
fi
