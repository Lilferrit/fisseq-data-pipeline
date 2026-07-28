nextflow.enable.dsl = 2

// OVWT_GLOBAL: wraps python -m fisseq_data_pipeline.ovwt. Runs once across all batches' normalized
// cells (globs normalization/cells/*.parquet), gated by params.run_global.
// Always filtered against the ANOVA_BLOCKLIST output -- there is no
// unfiltered global OvWT run. Publishes results.parquet and models.pkl
// under ovwt_global/.
process OVWT_GLOBAL {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/ovwt_global", mode: 'copy'

    input:
    tuple val(input_dir), val(feature_block_list_file)

    output:
    path("results.parquet")
    path("models.pkl")

    script:
    // TODO: add global OvWT visualization
    """
    echo "Starting OVWT_GLOBAL for global"
    python -m fisseq_data_pipeline.ovwt \\
        output_dir=. \\
        "input_file=${input_dir}/normalization/cells/*.parquet" \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        feature_block_list_file=${feature_block_list_file}
    """
}
