#!/bin/bash

TARGET_DIR="opt"

for file in "$TARGET_DIR"/rirs*; do
    if [ -f "$file" ] ; then
        zip "optzip/${file}.zip" "$file"
    fi
done