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
 *        ├──► BATCHVSBATCH (pre)      (per active global channel, scoped │
 *        │                             to that channel's batches)        │
 *        ├──► BATCH_CORRECT_FIT/TRANSFORM (per active global channel)    │
 *        │        └──► ANOVA (batch-corrected) (per active global channel) │
 *        ▼                                                                │
 *   NORMALIZE  (per batch)                                                │
 *        │                                                                │
 *        ├──► BATCHVSBATCH (post)     (per active global channel)        │
 *        ├──► ANOVA (normalized)      (per active global channel)        │
 *        │        └──► ANOVA_BLOCKLIST (per active global channel)       │
 *        ├──► OVWT_BATCHWISE          (per batch: unfiltered and         │
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
 *        ├──► OVWT_GLOBAL             (per active global channel, filtered│
 *        │                             against that channel's ANOVA_BLOCKLIST)│
 *        ├──► FEATURE_SELECT_BATCHWISE (per batch) ◄─────────────────────┘
 *        └──► FEATURE_SELECT_GLOBAL   (per active global channel)
 *
 * Batches join a global channel via that batch's YAML `global_channel` key
 * (string or list of strings); a batch naming no channel never contributes
 * to any global run. params.global_channels (default null) lists which
 * named channels actually run -- each gets its own BATCHVSBATCH/OVWT_GLOBAL/
 * ANOVA/BATCH_CORRECT_FIT+TRANSFORM/FEATURE_SELECT_GLOBAL, scoped to only
 * that channel's batches. A batch in zero active channels is skipped by the
 * entire global chain; a batch in multiple channels runs once per channel,
 * independently. See docs/configuration.md.
 *
 * Output layout:
 *   {pipeline_dir}/
 *     configs/                         *.yaml, one per batch (mandatory input)
 *     input/                           {batch_stem}.parquet (INPUT output)
 *     qc_filter/{batch_stem}/          filtered_cells, barcode_counts, summary TSV
 *     normalization/cells/             {batch_stem}.parquet
 *     normalization/normalizers/       {batch_stem}.normalizer.parquet
 *     ovwt_batchwise/{batch_stem}/     results.csv (enriched), models.pkl  (unfiltered)
 *     ovwt_batchwise_barcode_filtered/{batch_stem}/  same, filtered against BARCODE_BLOCKLIST
 *     ovwt_cellscores_batchwise/{batch_stem}/  cell_scores.parquet
 *     check_barcodes/{batch_stem}/     results.parquet  (per-variant Tukey HSD across barcodes)
 *     barcode_blocklist/{batch_stem}/  barcode_blocklist.parquet  (per-barcode median p_adj + barcode_ok)
 *     feature_select_batchwise/{batch_stem}/  {batch_stem}.parquet, feature_correlations
 *     global/{channel}/
 *       qc_filter_cells/, normalization_cells/   per-channel staged cell copies
 *       batchvsbatch/{pre,post}/         results.parquet
 *       anova/                           anova.parquet  (normalized cells)
 *       anova_blocklist/                 anova_blocklist.parquet
 *       ovwt_global/                     results.csv, models.pkl
 *       batch_correction/fit/            stats_vb.parquet, centroids.parquet
 *       batch_correction/cells/          {batch_stem}.parquet
 *       batch_correction/anova/          anova.parquet  (batch-corrected cells)
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
