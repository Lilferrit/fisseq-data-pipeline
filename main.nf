#!/usr/bin/env nextflow
// Single pipeline mode -- no --pipeline_mode dispatch needed unless/until a
// second mode is added.
//
// FISSEQ Data Pipeline -- Nextflow DSL2
//
// DAG:
//
//   params.yaml (experiments: [...]) ──► INPUT ──► input/<batch_stem>.parquet
//        │
//        ▼
//   QC_FILTER   (per experiment)
//        │
//        ▼
//   NORMALIZE   (per experiment)   ← z-score fit on WT control cells
//        │
//        ├──► OVWT_BATCHWISE  (per experiment; optional: params.run_ovwt)
//        │        └──► GLOBAL_OVWT  (per active global channel)
//        │
//        └──► Feature selection, batchwise (optional: params.run_feature_selection)
//                 └──► GLOBAL_FEATURE_SELECT  (per active global channel)
//
// Experiments join a global channel via their `global_channel` key in
// params.yaml (string or list of strings); an experiment naming no channel
// never contributes to a global run. params.global_channels (default null)
// lists which named channels actually run -- each gets its own GLOBAL_OVWT
// and GLOBAL_FEATURE_SELECT, scoped to only that channel's experiments.
// See docs/configuration.md.
//
// Output layout:
//   {pipeline_dir}/
//     input/{batch_stem}.parquet                     INPUT output
//     qc_filter/{batch_stem}/                        filtered_cells, barcode_counts, variants_per_barcode
//     normalization/cells/{batch_stem}.parquet
//     normalization/normalizers/{batch_stem}.normalizer.parquet
//     ovwt_batchwise/{batch_stem}/                   results, cell_scores, models.pkl
//     feature_select_batchwise/{batch_stem}/         aggregates, splits, correlations, blocklist, output
//     global/{chan}/ovwt_distinguishability/         global_scores.parquet
//     global/{chan}/feature_select/                  aggregate.parquet, blocklist.parquet
nextflow.enable.dsl = 2

include { FisseqPipeline } from './workflows/fisseq'

workflow {
    FisseqPipeline()
}
