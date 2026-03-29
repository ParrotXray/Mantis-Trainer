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
[ "$CLASSIFIER" = "true" ] && CMD="$CMD --classifier"
[ "$EXPORT" = "true" ] && CMD="$CMD --export"

exec $CMD "$@"
