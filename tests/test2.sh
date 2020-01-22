#!/usr/bin/env bash

## post main test of serialized data file for embeddings
FILE=mdl/foobar.pcl
if [ -f "$FILE" ]; then
    echo "$FILE exists"
else 
    echo "$FILE does not exists"
fi
