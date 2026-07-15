nextflow.enable.dsl = 2

// BATCHVSBATCH: wraps fisseq-batch-vs-batch. Parameterized over which cells
// glob, use_parent_name, and publish subdirectory to use, so
// workflows/fisseq.nf invokes this process twice via
// `include { BATCHVSBATCH as X }` aliasing: once pre-normalization
// (QC-filtered cells) and once post-normalization.
process BATCHVSBATCH {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/batchvsbatch/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(input_dir), val(cells_glob), val(use_parent_name), val(publish_subdir)

    output:
    path("results.parquet")

    script:
    """
    echo "Starting BATCHVSBATCH for ${publish_subdir}"
    fisseq-batch-vs-batch \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        use_parent_name=${use_parent_name} \\
        min_cells=${params.bvb_min_cells} \\
        min_batches=${params.bvb_min_batches}
    """
}
