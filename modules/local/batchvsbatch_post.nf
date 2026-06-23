nextflow.enable.dsl = 2

process BATCHVSBATCH_POST {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/batchvsbatch/post", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("results.parquet")

    script:
    """
    echo "Starting BATCHVSBATCH_POST for global"
    fisseq-batch-vs-batch \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        min_cells=${params.bvb_min_cells} \\
        min_batches=${params.bvb_min_batches}
    """
}
