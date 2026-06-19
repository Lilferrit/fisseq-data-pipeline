nextflow.enable.dsl = 2

process OVWT_GLOBAL {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/ovwt_global", mode: 'copy'

    input:
    val(input_dir)

    output:
    path("results.parquet")
    path("models.pkl")

    script:
    // TODO: add global OvWT visualization
    """
    echo "Starting OVWT_GLOBAL for global"
    fisseq-ovwt \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.downsample_wt}
    """
}
