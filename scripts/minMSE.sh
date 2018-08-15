#!/bin/bash
IN=$1

echo "Min Validation Values:"
grep "Testing MSE Full:" $IN | grep Val | sort -nk 7 | head

LOC=`grep "Testing MSE Full:" $IN | grep Val | sort -nk 7 | head -1 | awk -F'\t' '{print $2}'`

echo "Corresponding Test Value:"
grep "Testing MSE Full:" $IN | grep "$LOC"