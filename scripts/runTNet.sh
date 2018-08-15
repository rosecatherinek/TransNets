#!/bin/bash
DIR="/remote/curtis2/rkanjira/amazon_2017/www2018"
NAME=$1
LOGDIR="/usr1/public/rkanjira/RecoNN/www2018/logs"
export PYTHONPATH="/afs/cs.cmu.edu/user/rkanjira/GitHub/RecoNN/"
DATADIR="$DIR/data"
FMk=$2
FILTER=$3
DEV=$4  #GPUs

if [ -z "$NAME" ]
then
      echo "ERROR: Dataset Name is empty"
      exit 1
fi

if [ -z "$FMk" ]
then
	  echo "WARNING: FMk is empty, using default"
      FMk=8
fi

if [ -z "$FILTER" ]
then
	  echo "WARNING: FILTER is empty, using default"
      FILTER=100
fi

if [ ! -z "$DEV" ]
then
      export CUDA_VISIBLE_DEVICES=$DEV
fi

BATCH=10
MAXLEN=500
LR=0.0002
EPOCH=20
DIM=5
WINDOW=3



EMB="$DIR/emb/$NAME.w2v.pkl"
DICT="$DIR/emb/$NAME.dict.pkl"

ORIG="$DATADIR/INT_index_$NAME.train.dat.gz"

EPDIR="$DIR/epochs/$NAME"

MODOUT="$DIR/models/tnet/$NAME"
mkdir -p $MODOUT

LOG="$LOGDIR/tnet_${NAME}_${DIM}_${FMk}_${FILTER}.out"



echo "$BATCH $MAXLEN 64 $LR $EPOCH 0.5 $EMB $EPDIR/train $EPDIR/val $EPDIR/test $FILTER $DIM $DICT $ORIG 2 $FMk $WINDOW $MODOUT $LOG"

nohup python -m TNetMain $BATCH $MAXLEN 64 $LR $EPOCH 0.5 $EMB $EPDIR/train $EPDIR/val $EPDIR/test $FILTER $DIM $DICT $ORIG 2 $FMk $WINDOW "$MODOUT/" > $LOG &

 