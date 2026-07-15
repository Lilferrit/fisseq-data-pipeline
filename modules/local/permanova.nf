nextflow.enable.dsl = 2

// PERMANOVA: wraps fisseq-permanova. Parameterized over which cells glob and
// publish subdirectory to use, so workflows/fisseq.nf invokes this process
// twice via `include { PERMANOVA as X }` aliasing: once against normalized
// cells, once against batch-corrected cells.
process PERMANOVA {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(input_dir), val(cells_glob), val(publish_subdir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA for ${publish_subdir}"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        n_permutations=${params.permanova_n_permutations}
    """
}
