nextflow.enable.dsl = 2

// OVWT_BATCHWISE: wraps fisseq-ovwt. Runs once per batch on that batch's
// normalized cells, training a one-vs-wildtype XGBoost model per variant.
// Parameterized over block_list_file and publish_subdir so
// workflows/fisseq.nf invokes this process twice via
// `include { OVWT_BATCHWISE as X }` aliasing: once unfiltered
// (block_list_file=null, publish_subdir="ovwt_batchwise") and once filtered
// against the ANOVA_BLOCKLIST output (publish_subdir="ovwt_batchwise_filtered").
// block_list_file is a val (not a staged path) so a Groovy null can be
// passed through directly -- fisseq-ovwt's block_list_file=null CLI arg
// parses as Python None via Hydra/OmegaConf, the same trick used for
// downsample_fraction in qc_filter.nf.
// Publishes results.parquet, models.pkl, test_index.parquet, and
// train_index.parquet (consumed by OVWT_CELLSCORES_BATCHWISE, selected via
// params.single_cell_scores_source) under <publish_subdir>/<batch_stem>/.
process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.input_dir}/${publish_subdir}/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), val(block_list_file), val(publish_subdir)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl"), path("test_index.parquet"), path("train_index.parquet")

    script:
    // TODO: add per-batch OvWT visualization
    """
    echo "Starting OVWT_BATCHWISE for ${batch_stem} (${publish_subdir})"
    fisseq-ovwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.downsample_wt} \\
        block_list_file=${block_list_file}
    """
}
