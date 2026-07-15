nextflow.enable.dsl = 2

// cells_glob is a val (not staged into the task dir), so there is no
// staging collision here; output_root takes priority over output_dir in
// fisseq-aggregate-feature-type's own path resolution, so the output lands
// directly in the task work dir regardless of output_dir.
process AGGREGATE_FEATURE_TYPE {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.input_dir}/${publish_subdir}/aggregates" }, mode: 'copy'

    input:
    tuple val(batch_key), val(cells_glob), val(feature_type), val(publish_subdir)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    script:
    """
    echo "Starting AGGREGATE_FEATURE_TYPE for ${batch_key} / ${feature_type}"
    fisseq-aggregate-feature-type \\
        output_dir=. \\
        output_root=${feature_type} \\
        "input_file=${cells_glob}" \\
        aggregator=${feature_type} \\
        feature_batch_size=${params.feature_batch_size}
    mv ${feature_type}.*.parquet ${feature_type}.parquet
    """
}
