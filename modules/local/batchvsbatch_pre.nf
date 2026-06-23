nextflow.enable.dsl = 2

process BATCHVSBATCH_PRE {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/batchvsbatch/pre", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("results.parquet")

    script:
    """
    echo "Starting BATCHVSBATCH_PRE for global"
    fisseq-batch-vs-batch \\
        output_dir=. \\
        "input_file=${input_dir}/qc_filter/*/filtered_cells.parquet" \\
        use_parent_name=true \\
        min_cells=${params.bvb_min_cells} \\
        min_batches=${params.bvb_min_batches}
    """
}
