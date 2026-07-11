nextflow.enable.dsl = 2

process PERMANOVA_BATCH_CORRECTED {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/batch_correction/permanova", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA_BATCH_CORRECTED for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/batch_correction/cells/*.parquet" \\
        n_permutations=${params.permanova_n_permutations}
    """
}
