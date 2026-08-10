nextflow.enable.dsl = 2

// OVWTLOBO_BATCHWISE: wraps python -m fisseq_data_pipeline.ovwtlobo. Runs once per batch on
// that batch's normalized cells, independently of OVWT_BATCHWISE (no dependency on
// run_ovwt/ANOVA_BLOCKLIST/CHECK_BARCODES -- structurally an independent branch off
// norm_ch, like WTVWT_BATCHWISE/WTVVARIANTPOOL_BATCHWISE). For each non-wildtype
// variant with at least ovwt_lobo_min_barcodes_per_variant barcodes, holds out each
// of that variant's barcodes in turn, retrains an OvWT-equivalent binary classifier
// on the variant's remaining barcodes, and scores the held-out barcode -- testing
// whether OVWT_BATCHWISE's classifiers generalize to unseen barcodes rather than
// memorizing barcode-specific artifacts. Does not recompute OVWT_BATCHWISE's own
// (in-distribution) numbers; results.parquet is joinable to OVWT_BATCHWISE's own
// results.parquet on variant (and barcode) for a downstream generalization-gap
// computation. Gated per batch by params.run_ovwt_lobo, default FALSE (unlike
// run_wtvwt's default true) -- LOBO trains many more models per batch than a single
// OVWT_BATCHWISE pass. Reuses ovwt_min_cells/ovwt_downsample_wt/
// max_cells_per_barcode_wt/max_cells_per_barcode_variant from the OVWT_BATCHWISE
// section rather than duplicating them. feature_block_list_file/
// barcode_block_list_file follow OVWT_BATCHWISE's convention (both vals, so a
// Groovy null passes through as Python None) rather than WTVWT_BATCHWISE's (which
// doesn't wire them at all) -- LOBO is testing the same classifier OVWT_BATCHWISE
// trains, so it supports the same filters; this call site always passes both as
// null (unfiltered), matching OVWT_BATCHWISE_UNFILTERED's call, since LOBO isn't
// wired to consume ANOVA_BLOCKLIST/BARCODE_BLOCKLIST outputs.
// Publishes results.parquet and models.pkl under ovwtlobo_batchwise/<batch_stem>/.
process OVWTLOBO_BATCHWISE {
    errorStrategy 'ignore'
    publishDir { "${params.pipeline_dir}/ovwtlobo_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet), val(feature_block_list_file), val(barcode_block_list_file), \
          val(ovwt_min_cells), val(ovwt_lobo_min_cells_holdout), val(ovwt_lobo_min_barcodes_per_variant), \
          val(ovwt_downsample_wt), val(max_cells_per_barcode_wt), val(max_cells_per_barcode_variant)

    output:
    tuple val(batch_stem), path("results.parquet"), path("models.pkl")

    script:
    """
    echo "Starting OVWTLOBO_BATCHWISE for ${batch_stem}"
    python -m fisseq_data_pipeline.ovwtlobo \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        min_cells=${ovwt_min_cells} \\
        min_cells_holdout=${ovwt_lobo_min_cells_holdout} \\
        min_barcodes_per_variant=${ovwt_lobo_min_barcodes_per_variant} \\
        downsample_wt=${ovwt_downsample_wt} \\
        max_cells_per_barcode_wt=${max_cells_per_barcode_wt} \\
        max_cells_per_barcode_variant=${max_cells_per_barcode_variant} \\
        feature_block_list_file=${feature_block_list_file} \\
        barcode_block_list_file=${barcode_block_list_file}
    """
}
