#!/bin/bash -ue
echo "Starting QC_FILTER for batch1"
fisseq-qc-filter \
    output_dir=. \
    'cell_files=[batch1.parquet]' \
    bc_threshold=3 \
    variant_bc_threshold=3 \
    edit_distance_threshold=1
