nextflow.enable.dsl = 2

// Shells out to the same fisseq-aggregate-feature-type CLI as
// aggregate_feature_type.nf, with index_file set to one split half.
// index_file is staged into the task dir as half1.parquet/half2.parquet;
// the mv destination below is deliberately named differently so it never
// collides with (or relies on overwriting) that staged input.
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
        index_file=${index_file}
    mv ${feature_type}.*.parquet half${half_num}_agg.parquet
    """
}
