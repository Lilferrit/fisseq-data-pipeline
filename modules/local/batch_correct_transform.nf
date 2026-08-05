nextflow.enable.dsl = 2

// BATCH_CORRECT_TRANSFORM: wraps python -m fisseq_data_pipeline.batchcorrecttransform. Runs once
// per (channel, batch) pair -- one task for every batch that's a member of
// an active global channel, applying that channel's own BATCH_CORRECT_FIT
// statistics to rescale the batch's QC-filtered cells to the wildtype
// centroid. A batch belonging to multiple channels is corrected once per
// channel, independently, each publishing under its own channel's
// publish_subdir; a batch belonging to no active channel is skipped
// entirely for this chain. The channel identifier is named "chan" below --
// "channel" is a reserved Nextflow binding, see AGENTS.md.
process BATCH_CORRECT_TRANSFORM {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(chan), val(batch_stem), path(filtered_cells), path(stats_vb), path(centroids), val(publish_subdir)

    output:
    tuple val(chan), val(batch_stem), path("${batch_stem}.parquet"), emit: corrected

    script:
    """
    echo "Starting BATCH_CORRECT_TRANSFORM for ${batch_stem} (channel ${chan})"
    python -m fisseq_data_pipeline.batchcorrecttransform \\
        output_dir=. \\
        output_root=${batch_stem} \\
        input_file=${filtered_cells} \\
        batch=${batch_stem} \\
        stats_file=${stats_vb} \\
        centroids_file=${centroids} \\
        wt_label=WT
    mv ${batch_stem}.filtered_cells.parquet ${batch_stem}.parquet
    """
}
