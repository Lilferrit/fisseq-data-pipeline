#!/bin/bash -ue
echo "Starting OVWT_CELLSCORES_BATCHWISE for batch1"
fisseq-ovwt-cell-scores \
    output_dir=. \
    input_file=test_index.parquet \
    models_path=models.pkl
