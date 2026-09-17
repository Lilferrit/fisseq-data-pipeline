nextflow.enable.dsl = 2

// OVWT_BATCHWISE: wraps python -m fisseq_data_pipeline.ovwt. Runs once per
// experiment on that experiment's normalized cells.
//
// Per variant, this trains one one-vs-wildtype XGBoost model per
// cross-validation fold rather than a single held-out model, so every cell
// ends up with exactly one out-of-fold score. Each variant gets two
// distinguishability numbers: auroc_pooled (over all its cells at once) and
// auroc_median_barcode (per-barcode AUROC, medianed), the latter showing
// whether a variant's signal is broad-based across its barcodes or driven by
// one or two outliers.
//
// params.ovwt_cv_mode selects the fold scheme: "kfold" (params.ovwt_n_folds
// folds, every barcode present in every fold's training set) or
// "barcode_holdout" (whole barcodes held out of training, one barcode group
// per fold, so a variant with a single barcode is skipped). Under
// "barcode_holdout" params.ovwt_n_folds caps the fold count rather than fixing
// it, and null -- which Groovy renders as the literal `null` that Hydra parses
// back into None -- means one fold per barcode.
//
// Not aliased and not parameterized over block-lists or a publish subdir --
// the feature-filtered and barcode-filtered variants went away with
// ANOVA_BLOCKLIST and BARCODE_BLOCKLIST, and OVWT_GLOBAL was replaced by
// GLOBAL_OVWT, which aggregates these per-experiment scores instead of
// re-fitting on pooled cells.
process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    label 'process_high'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/ovwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(normalized_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("cell_scores.parquet"), path("models.pkl"), emit: ovwt

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting OVWT_BATCHWISE for ${batch_stem}"
    python -m fisseq_data_pipeline.ovwt \\
        output_dir=. \\
        input_file=${normalized_parquet} \\
        label_column=${params.filter_label_column} \\
        wt_label=${params.ovwt_wt_label} \\
        cv_mode=${params.ovwt_cv_mode} \\
        n_folds=${params.ovwt_n_folds} \\
        calibrate=${params.ovwt_calibrate} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        random_seed=${params.random_seed}
    """
}
