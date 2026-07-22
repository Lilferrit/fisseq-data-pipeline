nextflow.enable.dsl = 2

// Shells out to the same fisseq-aggregate-feature-type CLI as
// aggregate_feature_type.nf, with index_file set to one split half.
// index_file is staged into the task dir as half1.parquet/half2.parquet;
// the mv destination below is deliberately named differently so it never
// collides with (or relies on overwriting) that staged input.
// When params.aggregate_downsample_wt is set, the seed is derived from
// (bootstrap_idx, half_num) so every half of every bootstrap replicate draws
// an independent wildtype subsample -- this is what makes the bootstrap
// comparison actually test feature reproducibility against different WT
// samples, rather than reusing one fixed sample everywhere.
process AGGREGATE_HALF {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.input_dir}/${publish_subdir}/half_aggregates/bootstrap_${bootstrap_idx}/${feature_type}" }, mode: 'copy'

    input:
    tuple val(batch_key), val(bootstrap_idx), val(half_num), path(index_file), val(feature_type), val(cells_glob), val(publish_subdir)

    output:
    tuple val(batch_key), val(bootstrap_idx), val(feature_type), val(half_num), path("half${half_num}_agg.parquet")

    script:
    """
    echo "Starting AGGREGATE_HALF for ${batch_key} / bootstrap ${bootstrap_idx} / half ${half_num} / ${feature_type}"
    fisseq-aggregate-feature-type \\
        output_dir=. \\
        output_root=${feature_type} \\
        "input_file=${cells_glob}" \\
        aggregator=${feature_type} \\
        index_file=${index_file} \\
        downsample_wt=${params.aggregate_downsample_wt} \\
        seed=${(bootstrap_idx as int) * 2 + half_num}
    mv ${feature_type}.*.parquet half${half_num}_agg.parquet
    """
}
