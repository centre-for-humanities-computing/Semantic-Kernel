#!/usr/bin/env bash

# pipeline for Nucleus
## suppress warnings on call
start=`date +%s`
echo pipeline init
while true;do echo -n '>';sleep 1;done &

python src/semantic_vect.py #> log_corpus.txt
#python build_nucleus.py
#python -W ignore build_graph.py


kill $!; trap 'kill $!' SIGTERM
echo
echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime