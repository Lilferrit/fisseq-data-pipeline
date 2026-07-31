nextflow.enable.dsl = 2

// OVWT_GLOBAL: wraps python -m fisseq_data_pipeline.ovwt. Runs once per
// active global group (see workflows/fisseq.nf), over that group's
// normalized cells (globs cells_glob, a per-group directory published by
// STAGE_GROUP_CELLS). Always filtered against the ANOVA_BLOCKLIST output --
// there is no unfiltered global OvWT run. Publishes results.parquet and
// models.pkl under publish_subdir (e.g. "global/<group>/ovwt_global").
process OVWT_GLOBAL {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(cells_glob), val(feature_block_list_file), val(publish_subdir)

    output:
    path("results.parquet")
    path("models.pkl")

    script:
    // TODO: add global OvWT visualization
    """
    echo "Starting OVWT_GLOBAL for ${publish_subdir}"
    python -m fisseq_data_pipeline.ovwt \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        max_cells_per_barcode_wt=${params.max_cells_per_barcode_wt} \\
        max_cells_per_barcode_variant=${params.max_cells_per_barcode_variant} \\
        feature_block_list_file=${feature_block_list_file}
    """
}
