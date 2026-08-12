nextflow.enable.dsl = 2

// BATCH_CORRECT_FIT: wraps python -m fisseq_data_pipeline.batchcorrect. Runs once per active
// global channel (see workflows/fisseq.nf), waiting for all of that
// channel's STAGE_CHANNEL_QC batches, fitting per-(variant, batch)
// statistics and per-variant centroids across that channel's QC-filtered
// cells only. Emits stats_vb.parquet and centroids.parquet, consumed by
// BATCH_CORRECT_TRANSFORM. cells_files is a real `path` input -- that
// channel's flattened <batch_stem>.parquet files (like BATCHVSBATCH_PRE),
// collected via workflows/fisseq.nf's perChannelSignal rather than
// re-globbed from a published directory string -- so use_parent_name=false
// and Nextflow's -resume cache correctly tracks the actual file set (see
// anova.nf's comment for the general rationale). The channel identifier is
// named "chan" below -- "channel" is a reserved Nextflow binding, see
// AGENTS.md.
process BATCH_CORRECT_FIT {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), path(cells_files), val(publish_subdir)

    output:
    tuple val(chan), path("stats_vb.parquet"), path("centroids.parquet"), emit: fit_outputs

    script:
    """
    echo "Starting BATCH_CORRECT_FIT for ${publish_subdir}"
    python -m fisseq_data_pipeline.batchcorrect \\
        output_dir=. \\
        "input_file=./*.parquet" \\
        use_parent_name=false \\
        wt_label=WT
    """
}
