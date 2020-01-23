#!/usr/bin/env bash
for f in tests/*.sh;
    do
        bash "$f" -H
        echo 
done