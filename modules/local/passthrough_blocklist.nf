nextflow.enable.dsl = 2

// PASSTHROUGH_BLOCKLIST: wraps python -m fisseq_data_pipeline.passthroughblocklist.
// Feature-selection blocklist producer for feature types NOT in
// params.feature_select_wt_null_types (by default, the summary-statistic
// aggregators mean/median/MAD/std): no reproducibility computation, every
// feature in that feature type's full AGGREGATE_FEATURE_TYPE output is
// marked ok. A real process (rather than folded into AGGREGATE_FEATURE_TYPE)
// so its blocklist output has the exact same schema/shape as
// WT_NULL_BLOCKLIST's, letting both feed COMBINE_BLOCKLISTS unchanged.
process PASSTHROUGH_BLOCKLIST {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.pipeline_dir}/${publish_subdir}/blocklists" }, mode: 'copy'

    input:
    tuple val(batch_key), val(feature_type), path(aggregate_file), val(publish_subdir)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    script:
    """
    echo "Starting PASSTHROUGH_BLOCKLIST for ${batch_key} / ${feature_type}"
    python -m fisseq_data_pipeline.passthroughblocklist \\
        output_dir=. \\
        aggregate_file=${aggregate_file}
    mv blocklist.parquet ${feature_type}.parquet
    """
}
