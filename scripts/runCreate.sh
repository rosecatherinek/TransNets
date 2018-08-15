#!/bin/bash
DIR="/remote/curtis2/rkanjira/amazon_2017/www2018"
NAME=$1
export PYTHONPATH="/afs/cs.cmu.edu/user/rkanjira/GitHub/RecoNN/"
DATADIR="$DIR/data"
MAXLEN=$2 #1000 for large datasets  #change according to avg len

LOGDIR="/usr1/public/rkanjira/RecoNN/www2018/logs"
mkdir -p $LOGDIR

if [ -z "$NAME" ]
then
      echo "ERROR: Dataset Name is empty"
      exit 1
fi

if [ -z "$MAXLEN" ]
then
	  echo "WARNING: Max Len is empty, using default"
      MAXLEN=1000
fi

echo "Processing for dataset = $NAME"
echo "Max Length of pooled reviews set to $MAXLEN"
echo "Avg Length of reviews in the $NAME dataset train = "

gunzip -c "$DATADIR/index_$NAME.train.dat.gz" | awk -F'\t' '{print $4}' | awk '{print NF}' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'

INTYPE=( "train" "val" "test")


COL=3
DICT="$DIR/emb/$NAME.dict.pkl"

echo "Converting word to int representation for $NAME"

echo "$IN $COL $OUT_EMB $OUT_DICT $OUT_REV_DICT $EMB_SIZE $LOG"

for TYP in "${INTYPE[@]}"
do
	IN="$DATADIR/index_$NAME.$TYP.dat.gz"
	OUT="$DATADIR/INT_index_$NAME.$TYP.dat.gz"
	LOG="$LOGDIR/${NAME}_${TYP}_word2int.out"
	echo "create int files: $IN $OUT $DICT $COL $LOG"
	nohup python -m DatasetUtils.WordToIntMain $IN $OUT $DICT $COL > $LOG &
	
done

wait

echo "Creating user/item id shortlist"

DATFILE="$DATADIR/INT_index_$NAME.train.dat.gz"
UOUT="$DATADIR/${NAME}_UID.txt"
IOUT="$DATADIR/${NAME}_IID.txt"

gunzip -c $DATFILE | awk -F'\t' '{print $1}' | sort | uniq > $UOUT &
gunzip -c $DATFILE | awk -F'\t' '{print $2}' | sort | uniq > $IOUT &

wait

echo "Num Users in $NAME Train = "
wc -l $UOUT

echo "Num Items in $NAME Train = "
wc -l $IOUT

echo "Creating Epochs"

EPDIR="$DIR/epochs"
mkdir -p $EPDIR

for TYP in "${INTYPE[@]}"
do
	TYPEEPDIR="$EPDIR/$NAME/$TYP"
	mkdir -p $TYPEEPDIR
	for i in {1..5}
	do
		DATFILE="$DATADIR/INT_index_$NAME.train.dat.gz" #always train
		IN="$DATADIR/INT_index_$NAME.$TYP.dat.gz"
		OUT="$TYPEEPDIR/epoch_$i.gz"
		LOG="$LOGDIR/epoch_${NAME}_${TYP}_$i.out"
		echo "create epoch files: $DATFILE $IN $OUT $LOG"
		nohup python -m StreamingUtils.ToDisk $DATFILE $IN $OUT 100 $MAXLEN 10 1.0 > $LOG &
		if [ "$TYP" != "train" ]; then
			break  #only 1 epoch for val & test
		fi
	done
done

wait

echo "Avg Length of constructed epoch pooled reviews in the $NAME dataset train = "
TYPEEPDIR="$EPDIR/$NAME/train"
gunzip -c "$TYPEEPDIR/epoch_1.gz" | awk -F'\t' '{print $4}' | awk '{print NF}' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'

echo "$NAME Done"



