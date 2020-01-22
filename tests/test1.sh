#!/usr/bin/env bash

FILE=mdl/embeddings.pcl
if [ -f "$FILE" ]; then
    echo "$FILE exists"
else 
    echo "$FILE does not exists"
fi
