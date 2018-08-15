#!/bin/bash
DIR="/remote/curtis2/rkanjira/amazon_2017/www2018"
NAME=$1
LOGDIR="/usr1/public/rkanjira/RecoNN/www2018/logs"
export PYTHONPATH="/afs/cs.cmu.edu/user/rkanjira/GitHub/RecoNN/"
DATADIR="$DIR/data"
DEV=$2  #GPUs

if [ ! -z "$DEV" ]
then
      export CUDA_VISIBLE_DEVICES=$DEV
fi

BATCH=10
MAXLEN=500
LR=0.0002
EPOCH=20
DIM=5
FMk=5


EMB="$DIR/emb/$NAME.w2v.pkl"
DICT="$DIR/emb/$NAME.dict.pkl"

ORIG="$DATADIR/INT_index_$NAME.train.dat.gz"

EPDIR="$DIR/epochs/$NAME"

UOUT="$DATADIR/${NAME}_UID.txt"
IOUT="$DATADIR/${NAME}_IID.txt"

LOG="$LOGDIR/tnet_ext_${NAME}_${DIM}_${FMk}.out"


echo "$BATCH $MAXLEN 64 $LR $EPOCH 0.5 $EMB $EPDIR/train $EPDIR/val $EPDIR/test 100 $DIM $DICT $UOUT $IOUT $ORIG $FMk $LOG"

nohup python -m TNetExtMain $BATCH $MAXLEN 64 $LR $EPOCH 0.5 $EMB $EPDIR/train $EPDIR/val $EPDIR/test 100 $DIM $DICT $UOUT $IOUT $ORIG $FMk > $LOG & 

