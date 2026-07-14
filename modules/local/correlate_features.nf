nextflow.enable.dsl = 2

process CORRELATE_FEATURES {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/${publish_subdir}/correlations/${feature_type}" }, mode: 'copy'

    input:
    tuple val(batch_key), val(bootstrap_idx), val(feature_type), path(half1_agg), path(half2_agg), val(publish_subdir)

    output:
    tuple val(batch_key), val(feature_type), val(bootstrap_idx), path("bootstrap_${bootstrap_idx}.parquet")

    script:
    """
    echo "Starting CORRELATE_FEATURES for ${batch_key} / ${feature_type} / bootstrap ${bootstrap_idx}"
    fisseq-correlate-features \\
        output_dir=. \\
        half1_file=${half1_agg} \\
        half2_file=${half2_agg} \\
        label_column=meta_aa_changes
    mv correlations.parquet bootstrap_${bootstrap_idx}.parquet
    """
}
