nextflow.enable.dsl = 2

// Replaces feature_select_batchwise.nf and feature_select_global.nf.
process FINALIZE_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), path(feature_type_files), val(cells_glob), path(block_list_file), val(publish_subdir), \
          val(run_pca), val(pca_n_components), val(run_umap), val(umap_n_components), val(umap_n_neighbors), \
          val(umap_metric), val(umap_min_dist), val(umap_random_state)

    output:
    // pca_components.parquet only exists when run_pca=true
    tuple val(batch_key), path("output.parquet"), path("pca_components.parquet", optional: true)

    script:
    """
    echo "Starting FINALIZE_FEATURE_SELECT for ${batch_key}"
    mkdir -p ft
    mv ${feature_type_files} ft/
    python -m fisseq_data_pipeline.featureselect \\
        output_dir=. \\
        output_root=out \\
        "input_file=${cells_glob}" \\
        "feature_type_files=ft/*.parquet" \\
        block_list_file=${block_list_file} \\
        compute_impact_score=true \\
        run_pca=${run_pca} \\
        pca_n_components=${pca_n_components} \\
        run_umap=${run_umap} \\
        umap_n_components=${umap_n_components} \\
        umap_n_neighbors=${umap_n_neighbors} \\
        umap_metric=${umap_metric} \\
        umap_min_dist=${umap_min_dist} \\
        umap_random_state=${umap_random_state}
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
