#!/bin/bash

for file in $(ls ../opt/rirs-1500-*.json)
do
    inst=$(echo $file | cut -d '/' -f3 | cut -d '.' -f1)
    echo $inst
    python3 fast_py.py --data $file >> log/$inst.txt
done
