#!/bin/bash
#
# Scan a directory for Megatron-LM style blended data files (.idx + .bin pairs)
# and output a weighted data path string suitable for --data_paths argument.
#
# Usage:
#   scan_data_paths.sh <data_dir> [weight]
#
# Examples:
#   scan_data_paths.sh /data/tokenized_qwen3
#   scan_data_paths.sh /data/tokenized_qwen3 0.5
#
# Output example:
#   1.0 /data/tokenized_qwen3/data1_text_document 1.0 /data/tokenized_qwen3/data2_text_document

DATA_DIR="${1:?Usage: $0 <data_dir> [weight]}"
WEIGHT="${2:-1.0}"

result=""

# Collect all .idx files, strip the .idx suffix to get the data prefix name
for idx_file in "${DATA_DIR}"/*.idx; do
    # Skip if no .idx files found (glob returned the pattern itself)
    [ -e "${idx_file}" ] || continue

    # Extract the data name by removing .idx extension
    data_name="${idx_file%.idx}"

    # Check that the matching .bin file exists
    bin_file="${data_name}.bin"
    if [ -f "${bin_file}" ]; then
        # Append weight and full data path
        if [ -z "${result}" ]; then
            result="${WEIGHT} ${data_name}"
        else
            result="${result} ${WEIGHT} ${data_name}"
        fi
    else
        echo "Warning: missing .bin file for ${idx_file}, skipping" >&2
    fi
done

if [ -z "${result}" ]; then
    echo "Error: no valid data (.idx + .bin pairs) found in ${DATA_DIR}" >&2
    exit 1
fi

echo "${result}"