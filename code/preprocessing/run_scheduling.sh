#!/bin/bash

show_help() {
    echo "Usage: run.sh <end date>";
    echo ""
    echo "Arguments:"
    echo "  <end_date>         End date for the statistics (format: YYYY-MM-DD)."
    echo ""
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

END_DATE=$1

mkdir -p $RESULTS_DIR/scheduling
/app/scheduling/main --end_date $END_DATE --outputs-dir $RESULTS_DIR/scheduling --inputs-dir $DATA_DIR/scheduling
