#!/bin/bash
#the input file has additional cols
for fname in *.dat.gz; do
	echo $fname
	OUT="new_$fname"
	gunzip -c $fname |  awk -F'\t' {'printf "%s\t%s\t%s\t%s\n", $1, $3, $5, $8'}  | gzip > $OUT
	mv $OUT $fname
done