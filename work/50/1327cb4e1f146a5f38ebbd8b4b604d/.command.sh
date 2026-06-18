#!/bin/bash -ue
echo "Starting OVWT_BATCHWISE for batch1"
fisseq-ovwt \
    output_dir=. \
    input_file=filtered_cells.parquet \
    min_cells=25 \
    downsample_wt=50
