#!/bin/bash

for file in $(ls ./solutions/rirs*.json)
do
    inst=$(echo $file | cut -d '/' -f3 | cut -d '.' -f1)
    echo $inst
    python3 rev_path.py --data $file >> log/$inst.txt
done
