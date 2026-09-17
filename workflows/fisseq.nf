nextflow.enable.dsl = 2

// FisseqPipeline: the single end-to-end DAG.
//
//   INPUT -> QC_FILTER -> NORMALIZE
//                            |
//                            +-> OVWT_BATCHWISE (per experiment, gated by
//                            |     params.run_ovwt)
//                            |     `-> GLOBAL_OVWT (once per active global
//                            |           channel: per-experiment synonymous
//                            |           z-score of both AUROC columns, then
//                            |           cross-experiment median)
//                            |
//                            `-> bootstrap feature selection (per experiment,
//                                  gated by params.run_feature_selection)
//                                  `-> GLOBAL_FEATURE_SELECT (once per active
//                                        global channel)
//
// Experiments are declared as a list of maps under `experiments:` in
// params.yaml (loaded with -params-file); there is no <pipeline_dir>/configs/
// directory and no per-experiment override of arbitrary pipeline params. Each
// entry carries only batch_stem, input_paths, an optional global_channel, and
// the three optional INPUT-stage fields, which fall back to their
// pipeline-wide defaults when omitted. Every other parameter -- including
// every run gate and the single params.random_seed -- is pipeline-wide.
//
// Both global stages run once per named channel in params.global_channels
// (default null = none run), scoped to only the experiments whose
// `global_channel` key names that channel. See docs/configuration.md.

include { INPUT                  } from '../modules/local/input'
include { QC_FILTER              } from '../modules/local/qc_filter'
include { NORMALIZE              } from '../modules/local/normalize'
include { OVWT_BATCHWISE         } from '../modules/local/ovwt_batchwise'
include { GLOBAL_OVWT            } from '../modules/local/global_ovwt'
include { AGGREGATE_FEATURE_TYPE  as AGGREGATE_FEATURE_TYPE_BATCHWISE  } from '../modules/local/aggregate_feature_type'
include { GENERATE_SPLIT          as GENERATE_SPLIT_BATCHWISE          } from '../modules/local/generate_split'
include { AGGREGATE_HALF          as AGGREGATE_HALF_BATCHWISE          } from '../modules/local/aggregate_half'
include { CORRELATE_FEATURES      as CORRELATE_FEATURES_BATCHWISE      } from '../modules/local/correlate_features'
include { BLOCKLIST               as BLOCKLIST_BATCHWISE               } from '../modules/local/blocklist'
include { COMBINE_BLOCKLISTS      as COMBINE_BLOCKLISTS_BATCHWISE      } from '../modules/local/combine_blocklists'
include { FINALIZE_FEATURE_SELECT as FINALIZE_FEATURE_SELECT_BATCHWISE } from '../modules/local/finalize_feature_select'
include { GLOBAL_FEATURE_SELECT   } from '../modules/local/global_feature_select'

// The only keys an `experiments:` entry may carry. Anything else is a typo
// or an attempt to set a pipeline-wide param per experiment -- both are
// rejected with a clear message rather than silently ignored. A function, not
// a top-level `def` binding: DSL2 forbids bare statements at script scope.
def experimentKeys() {
    [
        'batch_stem',
        'input_paths',
        'global_channel',
        'feature_allowlist_file',
        'feature_blocklist_file',
        'csv_schema_scan_rows',
    ] as Set
}

// Every key of aggregate.py's _AGGREGATORS registry -- the only legal entries
// of params.feature_select_types. Kept in step with the Python side by
// tests/unit/test_nextflow_params.py, which parses this function and asserts it
// matches _AGGREGATORS exactly.
//
// This list has to exist here because the value is interpolated straight into
// AGGREGATE_FEATURE_TYPE / AGGREGATE_HALF's shell script. aggregate() does
// raise a clear "Unknown aggregator" ValueError, but that code never runs: a
// malformed entry (a stray quote from a mis-quoted params.yaml list, say)
// breaks the generated .command.sh at the bash level first, and
// errorStrategy 'ignore' then swallows the task. Validating here fails the
// whole run, before a single task is submitted, with a message that names the
// offending value.
def aggregatorKeys() {
    [
        'mean',
        'median',
        'MAD',
        'std',
        'KS',
        'signedKS',
        'QQ',
        'AUROC',
        'KSnegLogP',
        'AUROCnegLogP',
    ] as Set
}

// Every accepted value of params.ovwt_cv_mode, mirroring ovwt.py's CV_MODES.
def ovwtCvModes() {
    ['kfold', 'leave_one_barcode_out'] as Set
}

// Nextflow CLI overrides (--run_ovwt false) arrive as the Groovy-truthy
// String "false", so every gate must be coerced before use. Replaces
// lib/BatchParams.groovy's asBool(), which went away with per-batch overrides.
def asBool(v) {
    v == null ? false : v.toString().toBoolean()
}

workflow FisseqPipeline {
    // -params-file params.yaml is mandatory (nextflow.config carries no
    // parameter defaults), so fail fast with a specific message for every
    // required-with-no-default param rather than letting Nextflow's generic
    // "no such property" surface first.
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required.\n  Usage: nextflow run . --pipeline_dir /path/to/data -params-file params.yaml"
    }
    if (!(params.experiments instanceof List) || params.experiments.isEmpty()) {
        error "ERROR: params.experiments must be a non-empty list of experiment maps (see params.yaml)."
    }
    // Only meaningful when the feature-selection chain actually runs, but
    // checked before any of it is wired up.
    if (asBool(params.run_feature_selection)) {
        if (!(params.feature_select_types instanceof List) || params.feature_select_types.isEmpty()) {
            error "ERROR: params.feature_select_types must be a non-empty list of aggregator names. " +
                  "Valid names: ${aggregatorKeys().sort().join(', ')}."
        }
        def badTypes = params.feature_select_types.findAll { t -> !aggregatorKeys().contains(t) }
        if (badTypes) {
            error "ERROR: params.feature_select_types has unrecognized entry/entries: " +
                  "${badTypes.join(' | ')}. Valid names: ${aggregatorKeys().sort().join(', ')}. " +
                  "(A stray quote in one of those usually means a mis-quoted YAML list -- " +
                  "[median\", \"KS\"] instead of [\"median\", \"KS\"].)"
        }
    }
    if (!ovwtCvModes().contains(params.ovwt_cv_mode)) {
        error "ERROR: params.ovwt_cv_mode must be one of ${ovwtCvModes().sort().join(', ')}, " +
              "got '${params.ovwt_cv_mode}'."
    }

    // Validate and resolve every experiment map once, here, at
    // workflow-construction time. Global defaults fill in only the keys an
    // entry omits; an entry's own value always wins.
    def allowedKeys = experimentKeys()
    def resolvedExperiments = [:]
    params.experiments.eachWithIndex { entry, i ->
        if (!(entry instanceof Map)) {
            error "ERROR: params.experiments[${i}] must be a map, got ${entry?.getClass()?.simpleName}."
        }
        if (!(entry.batch_stem instanceof String) || entry.batch_stem.trim().isEmpty()) {
            error "ERROR: params.experiments[${i}] is missing a required, non-empty 'batch_stem' field."
        }
        def unknown = entry.keySet() - allowedKeys
        if (unknown) {
            error "ERROR: params.experiments[${i}] ('${entry.batch_stem}') has unrecognized key(s): " +
                  "${unknown.sort().join(', ')}. An experiment entry may only set " +
                  "${allowedKeys.sort().join(', ')} -- every other parameter is pipeline-wide, " +
                  "set it at the top level of params.yaml instead."
        }
        if (!(entry.input_paths instanceof List) || entry.input_paths.isEmpty()) {
            error "ERROR: params.experiments[${i}] ('${entry.batch_stem}') is missing a required, " +
                  "non-empty 'input_paths' list."
        }
        // global_channel accepts a bare String or a list of Strings; both
        // normalize to a list so membership tests below are uniform.
        def chans = entry.global_channel == null
            ? []
            : (entry.global_channel instanceof List ? entry.global_channel : [entry.global_channel])
        chans.each { c ->
            if (!(c instanceof String)) {
                error "ERROR: params.experiments[${i}] ('${entry.batch_stem}') global_channel entries " +
                      "must be strings, got ${c?.getClass()?.simpleName}."
            }
        }
        resolvedExperiments[entry.batch_stem] = [
            input_paths           : entry.input_paths,
            global_channel        : chans,
            feature_allowlist_file: entry.containsKey('feature_allowlist_file') ? entry.feature_allowlist_file : params.feature_allowlist_file,
            feature_blocklist_file: entry.containsKey('feature_blocklist_file') ? entry.feature_blocklist_file : params.feature_blocklist_file,
            csv_schema_scan_rows  : entry.containsKey('csv_schema_scan_rows') ? entry.csv_schema_scan_rows : params.csv_schema_scan_rows,
        ]
    }
    def batch_stems = params.experiments.collect { experiment -> experiment.batch_stem }
    def duplicate_stems = batch_stems.findAll { s -> batch_stems.count(s) > 1 }.unique()
    if (duplicate_stems) {
        error "ERROR: params.experiments has duplicate batch_stem value(s): ${duplicate_stems.join(', ')}. " +
              "Every experiment's batch_stem must be unique."
    }

    // Resolve pipeline_dir to an absolute path so global process scripts can
    // glob published outputs. Relative paths (e.g. ".") break inside Nextflow
    // work directories.
    def pipeline_dir_abs = file(params.pipeline_dir).toAbsolutePath().toString()

    // Per-channel fan-out. params.global_channels lists which named channels
    // actually run the two global stages. If it is null/[] (the default),
    // channels_ch is empty and both stages simply run zero tasks -- no `if`
    // gate needed ("filter channels, don't if").
    //
    // NOTE: the per-channel identifier is bound as "chan" in every closure
    // below, never "channel" -- "channel" is a reserved Nextflow binding
    // (lowercase alias for the Channel class) and silently resolves to
    // `nextflow.Channel` itself if reused as a variable name, rather than
    // failing loudly. See AGENTS.md.
    def activeChannels = (params.global_channels ?: []) as List<String>
    channels_ch = channel.fromList(activeChannels)

    // Step 0: INPUT -- one input/<batch_stem>.parquet per experiment.
    config_ch = channel.fromList(resolvedExperiments.keySet() as List).map { batch_stem ->
        def cfg = resolvedExperiments[batch_stem]
        tuple(batch_stem, cfg.input_paths, cfg.feature_allowlist_file,
              cfg.feature_blocklist_file, cfg.csv_schema_scan_rows)
    }
    input_ch = INPUT(config_ch)

    // Step 1: QC filter (per experiment).
    qc_ch = QC_FILTER(input_ch).qc_outputs

    // Step 2: normalization (per experiment) -- z-score fit on wildtype cells.
    // qc_ch carries: (batch_stem, filtered_cells, barcode_counts, variants_per_barcode)
    norm_input_ch = qc_ch.map { batch_stem, fc, _bc, _vpb -> tuple(batch_stem, fc) }
    NORMALIZE(norm_input_ch)
    norm_ch = NORMALIZE.out.normalized  // tuple(batch_stem, normalized_parquet)

    // Step 3: OvWT -- per experiment, k-fold cross-validated one-vs-wildtype
    // scoring. Every cell gets exactly one out-of-fold score; each variant
    // gets a pooled AUROC and a median-of-per-barcode AUROC.
    if (asBool(params.run_ovwt)) {
        OVWT_BATCHWISE(norm_ch)
        ovwt_ch = OVWT_BATCHWISE.out.ovwt  // (batch_stem, results, cell_scores, models)

        // Step 3b: GLOBAL_OVWT -- once per active global channel. Collects
        // real path objects (not a glob string) so -resume invalidates
        // correctly when an upstream results.parquet changes; groupTuple
        // keeps batch_stems and results in matching order, which is the
        // contract reconstruct_staged_paths relies on.
        global_ovwt_input_ch = ovwt_ch
            .map { batch_stem, results, _scores, _models -> tuple(batch_stem, results) }
            .combine(channels_ch)
            .filter { batch_stem, _results, chan -> chan in resolvedExperiments[batch_stem].global_channel }
            .map { batch_stem, results, chan -> tuple(chan, batch_stem, results) }
            .groupTuple(by: 0)
            .map { chan, stems, results -> tuple(chan, results, stems) }
        GLOBAL_OVWT(global_ovwt_input_ch)
    }

    // Step 4: feature selection -- decomposed bootstrap + per-feature-type
    // pipeline.
    //   Stage 1:    per-feature-type full aggregation.
    //   Stage 2a-d: per-bootstrap split -> per-half aggregation -> correlation
    //               -> per-feature-type blocklist (gathered over bootstraps).
    //   Stage 3:    combine per-feature-type blocklists.
    //   Stage 4:    join stage-1 aggregates, apply combined blocklist,
    //               pycytominer select.
    if (asBool(params.run_feature_selection)) {
        feature_types_ch = channel.fromList(params.feature_select_types)
        // Explicit cast: CLI overrides (--feature_select_bootstrap_reps 3)
        // arrive as Strings and silently produce a bogus range if left
        // uncoerced in a Groovy IntRange.
        bootstrap_ch = channel.of(1..(params.feature_select_bootstrap_reps as int))

        // Stage 1: full per-feature-type aggregation, one task per
        // (experiment, feature_type).
        agg_input_ch = norm_ch
            .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
            .combine(feature_types_ch)
            .map { batch_stem, cells_glob, feature_type ->
                tuple(batch_stem, cells_glob, feature_type, "feature_select_batchwise/${batch_stem}")
            }
        AGGREGATE_FEATURE_TYPE_BATCHWISE(agg_input_ch)
        agg_ch = AGGREGATE_FEATURE_TYPE_BATCHWISE.out  // (batch_stem, feature_type, agg_file)

        // Stage 2a: one 50/50 split per (experiment, bootstrap replicate).
        split_input_ch = norm_ch
            .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
            .combine(bootstrap_ch)
            .map { batch_stem, cells_glob, bootstrap_idx ->
                tuple(batch_stem, cells_glob, bootstrap_idx, "feature_select_batchwise/${batch_stem}")
            }
        GENERATE_SPLIT_BATCHWISE(split_input_ch)
        split_ch = GENERATE_SPLIT_BATCHWISE.out  // (batch_stem, bootstrap_idx, half1, half2)

        // Stage 2b: expand each split into two per-half tuples, cross with
        // feature types, and re-attach the experiment's normalized cells via
        // .combine(norm_ch, by: 0) (keyed on batch_stem -- norm_ch has exactly
        // one entry per experiment, so this is a per-experiment broadcast,
        // not a fan-out).
        // NOTE: .join() is NOT a broadcast operator -- for a many-to-one key
        // relationship like this one it silently keeps only one match per key
        // and drops the rest, starving every downstream stage. Only use
        // .join() where both sides are already collapsed to exactly one item
        // per key (see the finalize-stage joins below).
        half_ch = split_ch.flatMap { batch_stem, bootstrap_idx, half1, half2 ->
            [
                tuple(batch_stem, bootstrap_idx, 1, half1),
                tuple(batch_stem, bootstrap_idx, 2, half2),
            ]
        }
        agg_half_input_ch = half_ch
            .combine(feature_types_ch)
            // (batch_stem, bootstrap_idx, half_num, index_file, feature_type)
            .combine(norm_ch, by: 0)
            // (batch_stem, bootstrap_idx, half_num, index_file, feature_type, normalized_parquet)
            .map { batch_stem, bootstrap_idx, half_num, index_file, feature_type, normalized_parquet ->
                tuple(batch_stem, bootstrap_idx, half_num, index_file, feature_type,
                      normalized_parquet.toString(), "feature_select_batchwise/${batch_stem}")
            }
        AGGREGATE_HALF_BATCHWISE(agg_half_input_ch)
        half_agg_ch = AGGREGATE_HALF_BATCHWISE.out
        // (batch_stem, bootstrap_idx, feature_type, half_num, half_agg_file)

        // Stage 2c: group by (batch_stem, bootstrap_idx, feature_type) --
        // exactly 2 per group -- pair by half_num (not arrival order) before
        // correlating.
        corr_input_ch = half_agg_ch
            .groupTuple(by: [0, 1, 2])
            .map { batch_stem, bootstrap_idx, feature_type, half_nums, half_files ->
                def pairs = [half_nums, half_files].transpose().sort { pair -> pair[0] }
                tuple(batch_stem, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1],
                      "feature_select_batchwise/${batch_stem}")
            }
        CORRELATE_FEATURES_BATCHWISE(corr_input_ch)
        corr_ch = CORRELATE_FEATURES_BATCHWISE.out  // (batch_stem, feature_type, bootstrap_idx, corr_file)

        // Stage 2d: group by (batch_stem, feature_type) -- gathers all
        // bootstrap replicates. THE one intentional synchronization point,
        // scoped to this stage only.
        blocklist_input_ch = corr_ch
            .map { batch_stem, feature_type, _bootstrap_idx, correlation_file ->
                tuple(batch_stem, feature_type, correlation_file)
            }
            .groupTuple(by: [0, 1])
            .map { batch_stem, feature_type, correlation_files ->
                tuple(batch_stem, feature_type, correlation_files,
                      "feature_select_batchwise/${batch_stem}")
            }
        BLOCKLIST_BATCHWISE(blocklist_input_ch)
        bl_ch = BLOCKLIST_BATCHWISE.out  // (batch_stem, feature_type, blocklist_file)

        // Stage 3: group by batch_stem -- gathers all feature types.
        combine_bl_input_ch = bl_ch
            .map { batch_stem, _feature_type, blocklist_file -> tuple(batch_stem, blocklist_file) }
            .groupTuple(by: 0)
            .map { batch_stem, blocklist_files ->
                tuple(batch_stem, blocklist_files, "feature_select_batchwise/${batch_stem}")
            }
        COMBINE_BLOCKLISTS_BATCHWISE(combine_bl_input_ch)
        combined_bl_ch = COMBINE_BLOCKLISTS_BATCHWISE.out  // (batch_stem, combined_blocklist_file)

        // Stage 4: group stage-1 output by batch_stem (all feature types'
        // full aggregates), join norm_ch (raw cells, for metadata), join
        // stage-3's combined blocklist.
        finalize_input_ch = agg_ch
            .map { batch_stem, _feature_type, agg_file -> tuple(batch_stem, agg_file) }
            .groupTuple(by: 0)
            .join(norm_ch)
            .join(combined_bl_ch)
            .map { batch_stem, agg_files, normalized_parquet, combined_bl_file ->
                tuple(batch_stem, agg_files, normalized_parquet.toString(), combined_bl_file,
                      "feature_select_batchwise/${batch_stem}")
            }
        FINALIZE_FEATURE_SELECT_BATCHWISE(finalize_input_ch)

        // GLOBAL_FEATURE_SELECT -- once per active global channel. It reuses
        // each member experiment's already-published BATCHWISE artifacts
        // (feature_select_batchwise/<stem>/{aggregates,blocklist.parquet})
        // directly off pipeline_dir, looping over batch_stems in Python, so
        // it needs no cell-level recompute and no staged cells.
        def batchesByChannel = activeChannels.collectEntries { chan ->
            [chan, resolvedExperiments.findAll { _batch_stem, cfg ->
                cfg.global_channel.contains(chan)
            }.keySet() as List]
        }
        batchesByChannel.each { chan, stems ->
            if (stems.isEmpty()) {
                log.warn "Global channel '${chan}' has no member experiments -- " +
                    "GLOBAL_FEATURE_SELECT will have nothing to read for this channel."
            }
        }

        // Single-element signal that fires once every experiment's BATCHWISE
        // feature selection has been published -- gated on combined_bl_ch,
        // the last batchwise feature-select artifact.
        feature_select_ready_signal = combined_bl_ch.map { batch_stem, _bl -> batch_stem }.collect()
            .map { _stems -> pipeline_dir_abs }

        global_fs_input_ch = channels_ch
            .combine(feature_select_ready_signal)
            .map { chan, d ->
                tuple(chan, batchesByChannel[chan], d, "global/${chan}/feature_select")
            }
        GLOBAL_FEATURE_SELECT(global_fs_input_ch)
    }
}
