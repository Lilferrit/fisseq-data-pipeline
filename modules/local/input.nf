nextflow.enable.dsl = 2

// INPUT: wraps python -m fisseq_data_pipeline.input. Runs once per YAML config file in
// params.yaml_config_dir (optional upstream stage), producing one input/-ready
// cell-level Parquet file from a variant-selection spec instead of a
// pre-staged raw batch file. Publishes into the same input/ directory
// pre-staged batches live in, so QC_FILTER treats both origins identically.
process INPUT {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.input_dir}/input" }, mode: 'copy'

    input:
    tuple val(name), path(yaml_config)

    output:
    tuple val(name), path("${name}.parquet")

    script:
    """
    echo "Starting INPUT for ${name}"
    python -m fisseq_data_pipeline.input \\
        output_dir=. \\
        output_root=${name} \\
        config_path=${yaml_config}
    mv ${name}.output.parquet ${name}.parquet
    """
}
