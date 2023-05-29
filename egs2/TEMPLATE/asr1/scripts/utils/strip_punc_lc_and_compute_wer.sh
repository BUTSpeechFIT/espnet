#!/usr/bin/env bash

set -e
set -u
set -o pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <ref.trn>"
    exit;
fi

ref=$1

dir=$(dirname ${ref})

hyp=${dir}/hyp.trn


cat ${ref} | awk -F"\t" '{print $1}' > ${ref}.tmp
cat ${ref} | awk -F"\t" '{print $2}' > ${ref}.ids

detokenizer.perl -q < ${ref}.tmp > ${ref}.tmp.detok
remove_punctuation.pl < ${ref}.tmp.detok > ${ref}.tmp.detok.rm
lowercase.perl < ${ref}.tmp.detok.rm > ${ref}.tmp.detok.rm.lc
tokenizer.perl -q < ${ref}.tmp.detok.rm.lc > ${ref}.tmp.detok.rm.lc.tok

paste -d"\t" ${ref}.tmp.detok.rm.lc.tok ${ref}.ids > ${ref}.lc.rm


cat ${hyp} | awk -F"\t" '{print $1}' > ${hyp}.tmp
cat ${hyp} | awk -F"\t" '{print $2}' > ${hyp}.ids

detokenizer.perl -q < ${hyp}.tmp > ${hyp}.tmp.detok
remove_punctuation.pl < ${hyp}.tmp.detok > ${hyp}.tmp.detok.rm
lowercase.perl < ${hyp}.tmp.detok.rm > ${hyp}.tmp.detok.rm.lc
tokenizer.perl -q < ${hyp}.tmp.detok.rm.lc > ${hyp}.tmp.detok.rm.lc.tok

paste -d"\t" ${hyp}.tmp.detok.rm.lc.tok ${hyp}.ids > ${hyp}.lc.rm


sclite -r ${ref}.lc.rm trn -h ${hyp}.lc.rm trn -i rm -o all stdout > ${dir}/result.lc.rm.txt


grep -e Avg -e SPKR -m 2 ${dir}/result.lc.rm.txt
