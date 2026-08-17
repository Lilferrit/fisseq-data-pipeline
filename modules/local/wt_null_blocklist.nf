nextflow.enable.dsl = 2

// WT_NULL_BLOCKLIST: wraps python -m fisseq_data_pipeline.wtnullblocklist.
// Feature-selection stage 3 for the WT-null bootstrap branch -- the one
// intentional cross-bootstrap synchronization point for these feature
// types: gathers every bootstrap replicate's WT_NULL_AGGREGATE output for
// one (batch, feature type), averages each feature's null value across
// bootstraps, and marks each feature ok/blocked by an upper Tukey fence
// over the batch's per-feature null-mean distribution.
process WT_NULL_BLOCKLIST {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.pipeline_dir}/${publish_subdir}/blocklists" }, mode: 'copy'

    input:
    tuple val(batch_key), val(feature_type), path(wt_null_files), val(publish_subdir), val(tukey_multiplier)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    script:
    """
    echo "Starting WT_NULL_BLOCKLIST for ${batch_key} / ${feature_type}"
    python -m fisseq_data_pipeline.wtnullblocklist \\
        output_dir=. \\
        "wt_null_files=*.parquet" \\
        tukey_multiplier=${tukey_multiplier}
    mv blocklist.parquet ${feature_type}.parquet
    """
}
