#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * FISSEQ Data Pipeline — Nextflow DSL2
 *
 * DAG:
 *
 *   input/*.parquet
 *        │
 *        ▼
 *   QC_FILTER  (per batch) ──────────────────────────────────────────────┐
 *        │                                                                │ barcode_counts
 *        ├──► BATCHVSBATCH (pre)      (global, waits for all QC_FILTER)  │
 *        ▼                                                                │
 *   NORMALIZE  (per batch)                                                │
 *        │                                                                │
 *        ├──► BATCHVSBATCH (post)     (global, waits for all NORMALIZE)  │
 *        ├──► OVWT_BATCHWISE          (per batch)                        │
 *        │        └──► OVWT_CELLSCORES_BATCHWISE  (per batch, optional:  │
 *        │                 params.run_single_cell_scores; always on in  │
 *        │                 OvwtPipeline)                                 │
 *        │                 └──► CHECK_BARCODES  (per batch, optional:    │
 *        │                          params.run_check_barcodes; implies   │
 *        │                          run_single_cell_scores)               │
 *        ├──► OVWT_GLOBAL             (global, waits for all NORMALIZE)  │
 *        ├──► FEATURE_SELECT_BATCHWISE (per batch) ◄─────────────────────┘
 *        └──► FEATURE_SELECT_GLOBAL   (global, waits for all NORMALIZE)
 *
 * Output layout:
 *   {input_dir}/
 *     qc_filter/{batch_stem}/          filtered_cells, barcode_counts, summary TSV
 *     normalization/cells/             {batch_stem}.parquet
 *     normalization/normalizers/       {batch_stem}.normalizer.parquet
 *     batchvsbatch/pre/                results.parquet  (pre batch correction)
 *     batchvsbatch/post/               results.parquet  (post batch correction)
 *     ovwt_batchwise/{batch_stem}/     results.csv (enriched), models.pkl
 *     ovwt_global/                     results.csv, models.pkl
 *     ovwt_cellscores_batchwise/{batch_stem}/  cell_scores.parquet
 *     check_barcodes/{batch_stem}/     results.parquet  (per-variant Tukey HSD across barcodes)
 *     feature_select_batchwise/{batch_stem}/  {batch_stem}.parquet, feature_correlations
 *     feature_select_global/           global.parquet, feature_correlations, redundancy-filtered
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
