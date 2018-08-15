#!/bin/bash
for fname in index_*.train.dat.gz; do
	echo $fname
	echo "Num Records:"
	gunzip -c "$fname" | wc -l
	echo "Avg Len of Text Field:"
	gunzip -c "$fname" | awk -F'\t' '{print $4}' | awk '{print NF}' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
done


