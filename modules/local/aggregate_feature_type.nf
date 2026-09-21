nextflow.enable.dsl = 2

// cells_glob is a val (not staged into the task dir), so there is no
// staging collision here; output_root takes priority over output_dir in
// python -m fisseq_data_pipeline.aggregatefeaturetype's own path resolution, so the output lands
// directly in the task work dir regardless of output_dir.
// This process runs once per (batch, feature_type) -- no repeated
// per-instance identity to vary a downsample seed by -- so seed is fixed
// when params.feature_select_downsample_wt is set.
process AGGREGATE_FEATURE_TYPE {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    // publish_subdir is the full destination, not a parent: the caller picks
    // "<batch>/aggregates" or "<batch>/passthrough_aggregates" so that one
    // process definition serves both params.feature_select_types and
    // params.feature_select_passthrough_types. Keeping the two directories
    // apart is load-bearing -- GLOBAL_FEATURE_SELECT globs
    // "<batch>/aggregates/*.parquet", and publishDir mode: 'copy' never
    // removes anything, so a shared directory would leak passthrough
    // aggregates into the global stage.
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), val(cells_glob), val(feature_type), val(publish_subdir)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting AGGREGATE_FEATURE_TYPE for ${batch_key} / ${feature_type}"
    python -m fisseq_data_pipeline.aggregatefeaturetype \\
        output_dir=. \\
        output_root=${feature_type} \\
        "input_file=${cells_glob}" \\
        aggregator=${feature_type} \\
        downsample_wt=${params.feature_select_downsample_wt} \\
        random_seed=${params.random_seed} \\
        feature_chunk_size=${params.aggregate_feature_chunk_size}
    mv ${feature_type}.*.parquet ${feature_type}.parquet
    """
}
