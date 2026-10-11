#!/bin/bash
set -e
cd src/

CMD="./main.py"

if [ -n "$DATASET" ]; then
    CMD="$CMD --set $DATASET"
fi

if [ -n "$DATAPATH" ]; then
    CMD="$CMD --path $DATAPATH"
fi

[ "$ALL" = "true" ] && CMD="$CMD --all"
[ "$DATAPREPROCESS" = "true" ] && CMD="$CMD --datapreprocess"
[ "$DEEPAUTOENCODER" = "true" ] && CMD="$CMD --deepautoencoder"
[ "$EXPORT" = "true" ] && CMD="$CMD --export"
[ "$FEATUREANALYSIS" = "true" ] && CMD="$CMD --feature-analysis"
if [ "$FINETUNE" = "true" ]; then
    CMD="$CMD --finetune"
elif [ -n "$FINETUNE" ] && [ "$FINETUNE" != "false" ]; then
    CMD="$CMD --finetune $FINETUNE"
fi
[ -n "$FINETUNE_SINCE" ] && CMD="$CMD --finetune-since $FINETUNE_SINCE"
[ -n "$RESUME" ] && CMD="$CMD --resume $RESUME"

exec $CMD "$@"
