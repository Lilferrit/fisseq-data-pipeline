nextflow.enable.dsl = 2

process PERMANOVA {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/permanova", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        n_permutations=${params.permanova_n_permutations}
    """
}
