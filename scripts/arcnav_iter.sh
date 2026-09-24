#!/bin/bash
# Snapshot arcnav/ and run one evaluation set from the snapshot so later edits do not leak into a running set.
# usage: scripts/arcnav_iter.sh NAME [GAMES] [MINUTES] [JOBS] [CONFIG]
set -e
ROOT=/home/hss/code/kaggle/2026/arc-prize-2026-arc-agi-3
NAME=$1; GAMES=${2:-ls20,vc33,m0r0,r11l,sb26,tn36}; MIN=${3:-20}; JOBS=${4:-3}; CFG=$5
SNAP=$ROOT/vendor/arcnav-runs/snap-$NAME
if [ ! -d $SNAP/arcnav ]; then mkdir -p $SNAP; cp -r $ROOT/arcnav $SNAP/arcnav; ln -s $ROOT/environment_files $SNAP/environment_files; fi
cd $SNAP && [ -f $SNAP/COMMIT ] || git -C $ROOT rev-parse --short HEAD > $SNAP/COMMIT
exec $ROOT/.venv/bin/python -c "import sys; sys.path.insert(0, '$SNAP'); from arcnav.runner import main; main()" \
  --games $GAMES --minutes $MIN --jobs $JOBS --out $ROOT/experiments/arcnav/results --tag $NAME ${CFG:+--config $CFG} \
  > $ROOT/vendor/arcnav-runs/$NAME.log 2>&1
