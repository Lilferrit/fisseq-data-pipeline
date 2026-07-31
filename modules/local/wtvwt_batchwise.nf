nextflow.enable.dsl = 2

// WTVWT_BATCHWISE: wraps python -m fisseq_data_pipeline.wtvwt. Runs once per batch on that batch's
// normalized cells, restricting to wildtype-labeled cells and training one
// binary XGBoost classifier per pair of wildtype barcodes to distinguish
// cells belonging to either barcode. Gated per batch by params.run_wtvwt
// (see workflows/fisseq.nf's batchGates()). Publishes results.parquet and
// models.pkl under wtvwt_batchwise/<batch_stem>/.
process WTVWT_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/wtvwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), val(wtvwt_min_cells_per_barcode)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl")

    script:
    """
    echo "Starting WTVWT_BATCHWISE for ${batch_stem}"
    python -m fisseq_data_pipeline.wtvwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells_per_barcode=${wtvwt_min_cells_per_barcode}
    """
}
