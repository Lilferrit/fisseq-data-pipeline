nextflow.enable.dsl = 2

// Replaces feature_select_batchwise.nf and feature_select_global.nf.
process FINALIZE_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), path(feature_type_files), path(passthrough_files, stageAs: 'pt/*'), val(cells_glob), path(block_list_file), val(publish_subdir)

    output:
    // pca_components.parquet only exists when run_pca=true -- must be its
    // own output statement, not a third element of the tuple below: Nextflow
    // (26.04.6) does not honor per-element `optional: true` on a path()
    // nested inside a multi-element tuple output (it still raises
    // MissingFileException, which -- combined with errorStrategy 'ignore'
    // above -- silently drops output.parquet too). Declared standalone like
    // this, the optional file behaves correctly.
    tuple val(batch_key), path("output.parquet")
    path("pca_components.parquet", optional: true)

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting FINALIZE_FEATURE_SELECT for ${batch_key}"
    mkdir -p ft
    mv ${feature_type_files} ft/
    # passthrough_files is staged straight into pt/ (stageAs above) rather than
    # moved, because the list is empty whenever params.feature_select_passthrough_types
    # is -- an `mv` with no arguments would fail the task. mkdir -p keeps the
    # glob well-formed in that case; featureselect.py treats a zero-match
    # passthrough glob as a warning, not an error.
    mkdir -p pt
    python -m fisseq_data_pipeline.featureselect \\
        output_dir=. \\
        output_root=out \\
        "input_file=${cells_glob}" \\
        "feature_type_files=ft/*.parquet" \\
        "passthrough_feature_type_files=pt/*.parquet" \\
        block_list_file=${block_list_file} \\
        compute_impact_score=true \\
        label_column=${params.filter_label_column} \\
        run_pca=${params.run_pca} \\
        pca_n_components=${params.pca_n_components} \\
        run_umap=${params.run_umap} \\
        umap_n_components=${params.umap_n_components} \\
        umap_n_neighbors=${params.umap_n_neighbors} \\
        umap_metric=${params.umap_metric} \\
        umap_min_dist=${params.umap_min_dist} \\
        random_seed=${params.random_seed}
    # Rename the PCA-components file by its known exact name *before* the
    # generic glob rename below -- once run_pca=true produces a second
    # out.*.parquet file (out.pca_components.parquet), the glob would
    # otherwise become ambiguous.
    if [ -f out.pca_components.parquet ]; then
        mv out.pca_components.parquet pca_components.parquet
    fi
    mv out.*.parquet output.parquet
    """
}
