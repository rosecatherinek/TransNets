#!/bin/bash
DIR="/remote/curtis2/rkanjira/amazon_2017/www2018"
NAME=$1

EPDIR="$DIR/epochs"
TYPEEPDIR="$EPDIR/$NAME/train"

gunzip -c "$TYPEEPDIR/epoch_1.gz" | awk -F'\t' '{print $4}' | awk '{print NF}' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'

