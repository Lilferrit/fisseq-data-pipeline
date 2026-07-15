nextflow.enable.dsl = 2

// GENERATE_SPLIT: wraps fisseq-generate-split. Feature-selection stage 2a —
// one stratified 50/50 pseudo-replicate split per (batch or global,
// bootstrap replicate), seeded by bootstrap_idx. Emits half1.parquet and
// half2.parquet, consumed by AGGREGATE_HALF.
process GENERATE_SPLIT {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/${publish_subdir}/splits/bootstrap_${bootstrap_idx}" }, mode: 'copy'

    input:
    tuple val(batch_key), val(cells_glob), val(bootstrap_idx), val(publish_subdir)

    output:
    tuple val(batch_key), val(bootstrap_idx), path("half1.parquet"), path("half2.parquet")

    script:
    """
    echo "Starting GENERATE_SPLIT for ${batch_key} / bootstrap ${bootstrap_idx}"
    fisseq-generate-split \\
        output_dir=. \\
        "input_file=${cells_glob}" \\
        random_state=${bootstrap_idx}
    """
}
