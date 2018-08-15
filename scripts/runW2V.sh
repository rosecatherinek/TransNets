#!/bin/bash
DIR="/remote/curtis2/rkanjira/amazon_2017/www2018"
NAME=$1
DATADIR="$DIR/data"

LOGDIR="/usr1/public/rkanjira/RecoNN/www2018/logs"

mkdir -p $LOGDIR

IN="$DATADIR/index_$NAME.train.dat.gz"
COL=3
OUT_EMB="$DIR/emb/$NAME.w2v.pkl"
OUT_DICT="$DIR/emb/$NAME.dict.pkl"
OUT_REV_DICT="$DIR/emb/$NAME.rev_dict.pkl"
EMB_SIZE=64
LOG="$LOGDIR/w2v_$NAME.out"

echo "$IN $COL $OUT_EMB $OUT_DICT $OUT_REV_DICT $EMB_SIZE $LOG"

nohup python /afs/cs.cmu.edu/user/rkanjira/GitHub/RecoNN/DatasetUtils/Word2VecBasicGZ.py $IN $COL $OUT_EMB $OUT_DICT $OUT_REV_DICT $EMB_SIZE > $LOG &