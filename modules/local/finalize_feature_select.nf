nextflow.enable.dsl = 2

// Replaces feature_select_batchwise.nf and feature_select_global.nf.
process FINALIZE_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.input_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), path(feature_type_files), val(cells_glob), path(block_list_file), val(publish_subdir)

    output:
    tuple val(batch_key), path("output.parquet")

    script:
    """
    echo "Starting FINALIZE_FEATURE_SELECT for ${batch_key}"
    mkdir -p ft
    mv ${feature_type_files} ft/
    fisseq-feature-select \\
        output_dir=. \\
        output_root=out \\
        "input_file=${cells_glob}" \\
        "feature_type_files=ft/*.parquet" \\
        block_list_file=${block_list_file} \\
        compute_impact_score=true
    mv out.*.parquet output.parquet
    """
}
