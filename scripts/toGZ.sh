#!/bin/bash
for fname in *.dat; do
	echo $fname
	gzip $fname
done