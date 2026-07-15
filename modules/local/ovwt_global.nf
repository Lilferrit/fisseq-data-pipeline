nextflow.enable.dsl = 2

// OVWT_GLOBAL: wraps fisseq-ovwt. Runs once across all batches' normalized
// cells (globs normalization/cells/*.parquet), gated by params.global.
// Publishes results.parquet and models.pkl under ovwt_global/.
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
