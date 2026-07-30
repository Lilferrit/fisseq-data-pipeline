nextflow.enable.dsl = 2

// BLOCKLIST: wraps python -m fisseq_data_pipeline.blocklist. Feature-selection stage 2d — the one
// intentional cross-bootstrap synchronization point: gathers every bootstrap
// replicate's CORRELATE_FEATURES output for one (batch or global, feature
// type), and marks each feature ok/blocked by its median correlation.
process BLOCKLIST {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/${publish_subdir}/blocklists" }, mode: 'copy'

    input:
    tuple val(batch_key), val(feature_type), path(correlation_files), val(publish_subdir), val(minimum_correlation)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    script:
    """
    echo "Starting BLOCKLIST for ${batch_key} / ${feature_type}"
    python -m fisseq_data_pipeline.blocklist \\
        output_dir=. \\
        "correlation_files=*.parquet" \\
        minimum_correlation=${minimum_correlation}
    mv blocklist.parquet ${feature_type}.parquet
    """
}
