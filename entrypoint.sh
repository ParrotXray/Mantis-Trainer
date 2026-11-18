#!/bin/bash
set -e
cd src/

if [ -n "$DATAPATH" ]; then
    export DATASET_ROOT="$DATAPATH"
fi

if [ "$API" = "true" ]; then
    exec ./server.py "$@"
fi

CMD=(./main.py)

[ "$ALL" = "true" ] && CMD+=(--all)
[ "$DATAPREPROCESS" = "true" ] && CMD+=(--datapreprocess)
[ "$DEEPAUTOENCODER" = "true" ] && CMD+=(--deepautoencoder)
[ "$FINETUNE" = "true" ] && CMD+=(--finetune)
[ "$SCORE" = "true" ] && CMD+=(--score)
[ "$FEATUREANALYSIS" = "true" ] && CMD+=(--feature-analysis)
[ "$EXPORT" = "true" ] && CMD+=(--export)
[ -n "$ARGS" ] && CMD+=(--args "$ARGS")

exec "${CMD[@]}" "$@"
