nextflow.enable.dsl = 2

// OVWT_BATCHWISE: wraps fisseq-ovwt. Runs once per batch on that batch's
// normalized cells, training a one-vs-wildtype XGBoost model per variant.
// Publishes results.parquet, models.pkl, and test_index.parquet (consumed by
// OVWT_CELLSCORES_BATCHWISE) under ovwt_batchwise/<batch_stem>/.
process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/ovwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl"), path("test_index.parquet")

    script:
    // TODO: add per-batch OvWT visualization
    """
    echo "Starting OVWT_BATCHWISE for ${batch_stem}"
    fisseq-ovwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.downsample_wt}
    """
}
