nextflow.enable.dsl = 2

process PERMANOVA_SYN {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/permanova/synonymous", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA_SYN for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        variant_class_filter=Synonymous \\
        n_bootstraps=${params.permanova_n_bootstraps} \\
        sample_size=${params.permanova_sample_size}
    """
}
