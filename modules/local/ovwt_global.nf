nextflow.enable.dsl = 2

// OVWT_GLOBAL: wraps python -m fisseq_data_pipeline.ovwt. Runs once per
// active global channel (see workflows/fisseq.nf), over that channel's
// normalized cells -- cells_files is a real `path` input (that channel's
// STAGE_CHANNEL_CELLS output, collected via workflows/fisseq.nf's
// perChannelSignal) rather than a directory-glob `val`, so Nextflow's
// -resume cache correctly tracks the actual file set (see anova.nf's
// comment for the general rationale). Always filtered against the
// ANOVA_BLOCKLIST output -- there is no unfiltered global OvWT run.
// Publishes results.parquet and models.pkl under publish_subdir (e.g.
// "global/<channel>/ovwt_global").
process OVWT_GLOBAL {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple path(cells_files), val(feature_block_list_file), val(publish_subdir)

    output:
    path("results.parquet")
    path("models.pkl")

    script:
    // TODO: add global OvWT visualization
    """
    echo "Starting OVWT_GLOBAL for ${publish_subdir}"
    python -m fisseq_data_pipeline.ovwt \\
        output_dir=. \\
        "input_file=./*.parquet" \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        max_cells_per_barcode_wt=${params.max_cells_per_barcode_wt} \\
        max_cells_per_barcode_variant=${params.max_cells_per_barcode_variant} \\
        min_cells_per_barcode=${params.ovwt_min_cells_per_barcode} \\
        feature_block_list_file=${feature_block_list_file}
    """
}
