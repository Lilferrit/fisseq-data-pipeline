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
 *        ▼                                                                │
 *   NORMALIZE  (per batch)                                                │
 *        │                                                                │
 *        ├──► PERMANOVA_WT            (global, waits for all NORMALIZE)  │
 *        ├──► PERMANOVA_SYN           (global, waits for all NORMALIZE)  │
 *        ├──► OVWT_BATCHWISE          (per batch)                        │
 *        ├──► OVWT_GLOBAL             (global, waits for all NORMALIZE)  │
 *        ├──► FEATURE_SELECT_BATCHWISE (per batch) ◄─────────────────────┘
 *        └──► FEATURE_SELECT_GLOBAL   (global, waits for all NORMALIZE)
 *
 * Output layout:
 *   {input_dir}/
 *     qc_filter/{batch_stem}/          filtered_cells, barcode_counts, summary TSV
 *     normalization/cells/             {batch_stem}.parquet
 *     normalization/normalizers/       {batch_stem}.normalizer.parquet
 *     permanova/wildtype/              permanova.parquet
 *     permanova/synonymous/            permanova.parquet
 *     ovwt_batchwise/{batch_stem}/     results.csv (enriched), models.pkl
 *     ovwt_global/                     results.csv, models.pkl
 *     feature_select_batchwise/{batch_stem}/  {batch_stem}.parquet, feature_correlations
 *     feature_select_global/           global.parquet, feature_correlations, redundancy-filtered
 */

// ── Parameters ───────────────────────────────────────────────────────────────

params.input_dir               = null
params.bc_threshold            = 10
params.variant_bc_threshold    = 4
params.edit_distance_threshold = 1
params.minimum_correlation     = 0.5
params.permanova_n_bootstraps  = 200
params.permanova_sample_size   = 1000
params.ovwt_min_cells          = 250
params.downsample_wt           = 5000
params.aggregator              = "multi"
params.workflow                = "fisseq"   // "fisseq" | "ovwt"

// ── Entry point ──────────────────────────────────────────────────────────────

include { FisseqPipeline } from './workflows/fisseq'
include { OvwtPipeline   } from './workflows/ovwt'

workflow {
    if (params.workflow == "ovwt") {
        OvwtPipeline()
    } else {
        FisseqPipeline()
    }
}
