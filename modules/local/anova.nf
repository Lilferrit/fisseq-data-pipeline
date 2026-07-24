nextflow.enable.dsl = 2

// ANOVA: wraps fisseq-anova. Parameterized over which cells glob and
// publish subdirectory to use, so workflows/fisseq.nf invokes this process
// twice via `include { ANOVA as X }` aliasing: once against normalized
// cells, once against batch-corrected cells.
process ANOVA {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(input_dir), val(cells_glob), val(publish_subdir)

    output:
    path("anova.parquet")

    script:
    """
    echo "Starting ANOVA for ${publish_subdir}"
    fisseq-anova \\
        output_dir=. \\
        "input_file=${cells_glob}"
    """
}
