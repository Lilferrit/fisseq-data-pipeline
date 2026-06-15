nextflow.enable.dsl = 2

process PERMANOVA_WT {
    errorStrategy 'terminate'
    publishDir "${params.input_dir}/permanova/wildtype", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("permanova.parquet")

    script:
    """
    echo "Starting PERMANOVA_WT for global"
    fisseq-permanova \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        variant_class_filter=WT \\
        n_bootstraps=${params.permanova_n_bootstraps} \\
        sample_size=${params.permanova_sample_size}
    """
}
