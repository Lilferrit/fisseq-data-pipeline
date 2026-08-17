nextflow.enable.dsl = 2

// WT_NULL_AGGREGATE: wraps python -m fisseq_data_pipeline.wtnullaggregate.
// Feature-selection stage 2 for the WT-null bootstrap branch (feature types
// in params.feature_select_wt_null_types only): one bootstrap replicate,
// splitting the batch's control pool into two disjoint halves seeded by
// bootstrap_idx, optionally downsampling each half independently
// (seed = bootstrap_idx*2 + half_num, so every half of every bootstrap
// replicate draws an independent wildtype subsample -- same seed idiom the
// old AGGREGATE_HALF used), and computing the configured aggregator
// between the two halves for every feature. Feeds into WT_NULL_BLOCKLIST,
// which gathers every bootstrap replicate and applies the Tukey-fence gate.
process WT_NULL_AGGREGATE {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}/wt_null/bootstrap_${bootstrap_idx}/${feature_type}" }, mode: 'copy'

    input:
    tuple val(batch_key), val(feature_type), val(bootstrap_idx), val(cells_glob), val(publish_subdir), val(downsample_wt), val(per_barcode), val(barcode_column)

    output:
    tuple val(batch_key), val(feature_type), val(bootstrap_idx), path("bootstrap_${bootstrap_idx}.parquet")

    script:
    """
    echo "Starting WT_NULL_AGGREGATE for ${batch_key} / ${feature_type} / bootstrap ${bootstrap_idx}"
    python -m fisseq_data_pipeline.wtnullaggregate \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        aggregator=${feature_type} \\
        bootstrap_idx=${bootstrap_idx} \\
        downsample_wt=${downsample_wt} \\
        per_barcode=${per_barcode} \\
        barcode_column=${barcode_column}
    # Unique per-bootstrap filename -- WT_NULL_BLOCKLIST gathers every
    # bootstrap replicate's output into one task dir via a glob, and every
    # replicate would otherwise write the same "wt_null.parquet" basename
    # (same collision WT_NULL_BLOCKLIST's old CORRELATE_FEATURES predecessor
    # avoided the same way).
    mv wt_null.parquet bootstrap_${bootstrap_idx}.parquet
    """
}
