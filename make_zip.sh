#!/bin/bash

TARGET_DIR="opt"

for file in "$TARGET_DIR"/*; do
    # 일반 파일만 처리
    if [ -f "$file" ]; then
        zip "optzip/${file}.zip" "$file"
    fi
done