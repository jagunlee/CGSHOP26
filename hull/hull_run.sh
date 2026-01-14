#!/bin/bash

for file in $(ls ../data/benchmark_instances/*.json)
do
    inst=$(echo $file | cut -d '/' -f4 | cut -d '.' -f1)
    echo $inst
    python3 draw.py --data $file
done
