#!/bin/bash

# unset -v START_DATE
unset -v END_DATE
unset -v OUTPUTS_DIR
unset -v INPUTS_DIR

OUTPUTS_DIR="outputs"
INPUTS_DIR="inputs"


show_help() {
    echo 'Usage: run.sh -o <outputs directory> -s <start date> -e <end date> -a <authorization token> -c  <compute scheduling> -m <compute monthly statistics> -u <download_params.json> -p <city_params.json> -f <table_fasciaOraria.json> -z <zone_params.json> -g <agent_params.json>';
    echo ""
    echo "Mandatory arguments:"
    echo "  -e <end_date>                   End date for the statistics (YYYY-MM-DD)."
    echo ""
    echo "Optional arguments:"
    echo "  --outputs-dir <path>            Directory for output results. Default: outputs."
    echo "  --inputs-dir <path>             Directory for input files. Default: inputs."
    echo ""
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# option parsing
while getopts o:e:i: opt; do
    case $opt in
            o) OUTPUTS_DIR=${OPTARG};;               # outputs directory (path to directory)
            e) END_DATE=${OPTARG};;                 # end date  (date format)          
            i) INPUTS_DIR=${OPTARG};;            # path to city_params.json
            *)
                    echo 'Error in command line parsing' >&2
                    exit 1
    esac
done

shift "$(( OPTIND - 1 ))"


missing_params=()

if [ -z "$END_DATE" ]; then
    missing_params+=("-e end_date")
fi

# Check if there are any missing parameters
if [ ${#missing_params[@]} -ne 0 ]; then
    echo "Error: The following mandatory parameters are missing:" >&2
    for param in "${missing_params[@]}"; do
        echo "  $param" >&2
    done
    show_help
    exit 1
fi

# run the main program
export PATH=$(pwd):$PATH

COMMAND="./main --end_date $END_DATE --outputs-dir $OUTPUTS_DIR --inputs_dir $INPUTS_DIR"

eval $COMMAND
