#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * FISSEQ Data Pipeline — Nextflow DSL2
 *
 * DAG:
 *
 *   configs/*.yaml (mandatory, one per batch) ──► INPUT ──► input/*.parquet
 *        │
 *        ▼
 *   QC_FILTER  (per batch) ──────────────────────────────────────────────┐
 *        │                                                                │ barcode_counts
 *        ├──► BATCHVSBATCH (pre)      (per active global group, scoped   │
 *        │                             to that group's batches)          │
 *        ▼                                                                │
 *   NORMALIZE  (per batch)                                                │
 *        │                                                                │
 *        ├──► BATCHVSBATCH (post)     (per active global group)          │
 *        ├──► OVWT_BATCHWISE          (per batch: unfiltered,            │
 *        │        │                    feature-filtered against          │
 *        │        │                    ANOVA_BLOCKLIST, and              │
 *        │        │                    barcode-filtered -- see below)    │
 *        │        └──► OVWT_CELLSCORES_BATCHWISE  (per batch, optional:  │
 *        │                 params.run_single_cell_scores; always on in  │
 *        │                 OvwtPipeline)                                 │
 *        │                 └──► CHECK_BARCODES  (per batch, optional:    │
 *        │                          params.run_check_barcodes; implies   │
 *        │                          run_single_cell_scores)               │
 *        │                          └──► BARCODE_BLOCKLIST  (per batch,  │
 *        │                                   optional: params.run_barcode_ │
 *        │                                   filtered_ovwt; implies      │
 *        │                                   run_check_barcodes)         │
 *        │                                   └──► OVWT_BATCHWISE          │
 *        │                                        (barcode-filtered pass)│
 *        ├──► OVWT_GLOBAL             (per active global group)          │
 *        ├──► FEATURE_SELECT_BATCHWISE (per batch) ◄─────────────────────┘
 *        └──► FEATURE_SELECT_GLOBAL   (per active global group)
 *
 * Batches join a global group via that batch's YAML `global_group` key
 * (string or list of strings); a batch naming no group never contributes to
 * any global run. params.global_groups (default null) lists which named
 * groups actually run -- each gets its own BATCHVSBATCH/OVWT_GLOBAL/
 * FEATURE_SELECT_GLOBAL, scoped to only that group's batches. See
 * docs/configuration.md.
 *
 * Output layout:
 *   {pipeline_dir}/
 *     configs/                         *.yaml, one per batch (mandatory input)
 *     input/                           {batch_stem}.parquet (INPUT output)
 *     qc_filter/{batch_stem}/          filtered_cells, barcode_counts, summary TSV
 *     normalization/cells/             {batch_stem}.parquet
 *     normalization/normalizers/       {batch_stem}.normalizer.parquet
 *     ovwt_batchwise/{batch_stem}/     results.csv (enriched), models.pkl  (unfiltered)
 *     ovwt_batchwise_feature_filtered/{batch_stem}/  same, filtered against ANOVA_BLOCKLIST
 *     ovwt_batchwise_barcode_filtered/{batch_stem}/  same, filtered against BARCODE_BLOCKLIST
 *     ovwt_cellscores_batchwise/{batch_stem}/  cell_scores.parquet
 *     check_barcodes/{batch_stem}/     results.parquet  (per-variant Tukey HSD across barcodes)
 *     barcode_blocklist/{batch_stem}/  barcode_blocklist.parquet  (per-barcode median p_adj + barcode_ok)
 *     feature_select_batchwise/{batch_stem}/  {batch_stem}.parquet, feature_correlations
 *     global/{group}/
 *       qc_filter_cells/, normalization_cells/   per-group staged cell copies
 *       batchvsbatch/{pre,post}/         results.parquet
 *       ovwt_global/                     results.csv, models.pkl
 *       feature_select/                  global.parquet, feature_correlations, redundancy-filtered
 */

// ── Entry point ──────────────────────────────────────────────────────────────

include { FisseqPipeline } from './workflows/fisseq'
include { OvwtPipeline   } from './workflows/ovwt'

workflow {
    if (params.pipeline_mode == "ovwt") {
        OvwtPipeline()
    } else {
        FisseqPipeline()
    }
}
