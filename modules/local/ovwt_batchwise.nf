nextflow.enable.dsl = 2

// OVWT_BATCHWISE: wraps python -m fisseq_data_pipeline.ovwt. Runs once per batch on that batch's
// normalized cells, training a one-vs-wildtype XGBoost model per variant.
// Parameterized over feature_block_list_file, barcode_block_list_file, and
// publish_subdir so workflows/fisseq.nf invokes this process three times via
// `include { OVWT_BATCHWISE as X }` aliasing: unfiltered
// (both block-list vals null, publish_subdir="ovwt_batchwise"),
// feature-filtered against the ANOVA_BLOCKLIST output
// (publish_subdir="ovwt_batchwise_feature_filtered"), and barcode-filtered
// against that batch's BARCODE_BLOCKLIST output
// (publish_subdir="ovwt_batchwise_barcode_filtered"). Both block-list vals
// are independent -- either, both, or neither may be non-null for a given
// call.
// Both block-list fields are vals (not staged paths) so a Groovy null can be
// passed through directly -- python -m fisseq_data_pipeline.ovwt's feature_block_list_file=null /
// barcode_block_list_file=null CLI args parse as Python None via
// Hydra/OmegaConf, the same trick used for qc_downsample_fraction in
// qc_filter.nf.
// Publishes results.parquet, models.pkl, test_index.parquet, and
// train_index.parquet (consumed by OVWT_CELLSCORES_BATCHWISE, selected via
// params.single_cell_scores_split) under <publish_subdir>/<batch_stem>/.
process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/${publish_subdir}/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), val(feature_block_list_file), val(barcode_block_list_file), val(publish_subdir)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl"), path("test_index.parquet"), path("train_index.parquet")

    script:
    // TODO: add per-batch OvWT visualization
    """
    echo "Starting OVWT_BATCHWISE for ${batch_stem} (${publish_subdir})"
    python -m fisseq_data_pipeline.ovwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        feature_block_list_file=${feature_block_list_file} \\
        barcode_block_list_file=${barcode_block_list_file}
    """
}
