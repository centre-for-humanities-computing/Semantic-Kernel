#!/usr/bin/env bash

# pipeline for Nucleus
## suppress warnings on call
start=`date +%s`
echo pipeline init
while true;do echo -n '>';sleep 1;done &

#python src/semantic_vect.py #> log_corpus.txt # &>/dev/null
#python src/nucleus_build.py Skydebanegade Mysundegade Enghave Plads Dannebrogsgade
python src/nucleus_build.py vartov bordel støj råb
python -W ignore src/graph_build.py


kill $!; trap 'kill $!' SIGTERM
echo
echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime