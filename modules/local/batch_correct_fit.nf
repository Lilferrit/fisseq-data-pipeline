nextflow.enable.dsl = 2

// BATCH_CORRECT_FIT: wraps python -m fisseq_data_pipeline.batchcorrect. Runs once per active
// global channel (see workflows/fisseq.nf), waiting for all of that
// channel's STAGE_CHANNEL_QC batches, fitting per-(variant, batch)
// statistics and per-variant centroids across that channel's QC-filtered
// cells only. Emits stats_vb.parquet and centroids.parquet, consumed by
// BATCH_CORRECT_TRANSFORM. cells_glob points at a channel-scoped, flattened
// <batch_stem>.parquet directory (like BATCHVSBATCH_PRE), so
// use_parent_name=false. The channel identifier is named "chan" below --
// "channel" is a reserved Nextflow binding, see AGENTS.md.
process BATCH_CORRECT_FIT {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(cells_glob), val(publish_subdir)

    output:
    tuple val(chan), path("stats_vb.parquet"), path("centroids.parquet"), emit: fit_outputs

    script:
    """
    echo "Starting BATCH_CORRECT_FIT for ${publish_subdir}"
    python -m fisseq_data_pipeline.batchcorrect \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        use_parent_name=false \\
        wt_label=WT
    """
}
