#!/usr/bin/env bash

python src/semantic_vect.py &>/dev/null
FILE=mdl/embeddings.pcl
if [ -f "$FILE" ]; then
    echo "$FILE exists"
    echo "> test 1 passed"
    rm mdl/embeddings.pcl
else 
    echo "$FILE does not exists"
    echo "> test 1 not passed"
fi
