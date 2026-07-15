nextflow.enable.dsl = 2

// BATCH_CORRECT_FIT: wraps fisseq-batch-correct-fit. Runs once globally,
// waiting for all QC_FILTER batches, fitting per-(variant, batch) statistics
// and per-variant centroids across all QC-filtered cells. Emits
// stats_vb.parquet and centroids.parquet, consumed by BATCH_CORRECT_TRANSFORM.
process BATCH_CORRECT_FIT {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/batch_correction/fit", mode: 'copy'

    input:
    val(input_dir)

    output:
    tuple path("stats_vb.parquet"), path("centroids.parquet"), emit: fit_outputs

    script:
    """
    echo "Starting BATCH_CORRECT_FIT for global"
    fisseq-batch-correct-fit \\
        output_dir=. \\
        "input_file=${input_dir}/qc_filter/*/filtered_cells.parquet" \\
        use_parent_name=true \\
        wt_label=WT
    """
}
