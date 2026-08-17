nextflow.enable.dsl = 2

// cells_glob is a val (not staged into the task dir), so there is no
// staging collision here; output_root takes priority over output_dir in
// python -m fisseq_data_pipeline.aggregatefeaturetype's own path resolution, so the output lands
// directly in the task work dir regardless of output_dir.
// This process runs once per (batch, feature_type) -- no repeated
// per-instance identity to vary a downsample seed by -- so seed is fixed
// when params.feature_select_downsample_wt is set. per_barcode/barcode_column
// are a plain flag/column name, not a randomization knob, so they need no
// per-instance variation either -- but they must match AGGREGATE_HALF's
// setting for the same batch, or the correlation stability check stops being
// apples-to-apples (both are resolved from the same
// resolvedBatchConfigs[batch_stem] entry in the workflow, so this is
// automatic).
process AGGREGATE_FEATURE_TYPE {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}/aggregates" }, mode: 'copy'

    input:
    tuple val(batch_key), val(cells_glob), val(feature_type), val(publish_subdir), val(downsample_wt), val(per_barcode), val(barcode_column)

    output:
    tuple val(batch_key), val(feature_type), path("${feature_type}.parquet")

    script:
    """
    echo "Starting AGGREGATE_FEATURE_TYPE for ${batch_key} / ${feature_type}"
    python -m fisseq_data_pipeline.aggregatefeaturetype \\
        output_dir=. \\
        output_root=${feature_type} \\
        "input_file=${cells_glob}" \\
        aggregator=${feature_type} \\
        downsample_wt=${downsample_wt} \\
        seed=0 \\
        per_barcode=${per_barcode} \\
        barcode_column=${barcode_column}
    mv ${feature_type}.*.parquet ${feature_type}.parquet
    """
}
