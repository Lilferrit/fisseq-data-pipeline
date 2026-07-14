nextflow.enable.dsl = 2

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
