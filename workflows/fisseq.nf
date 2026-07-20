nextflow.enable.dsl = 2

// FisseqPipeline: the default, full end-to-end DAG. Wires together QC_FILTER
// -> NORMALIZE -> BATCHVSBATCH (pre/post) -> OVWT (batchwise/global) ->
// bootstrap feature selection (batchwise/global, gated by params.global) ->
// BATCH_CORRECT_FIT/TRANSFORM -> PERMANOVA (normalized and batch-corrected).
// See AGENTS.md's "Project overview" DAG diagram for the full picture.
include { INPUT                     } from '../modules/local/input'
include { QC_FILTER                 } from '../modules/local/qc_filter'
include { NORMALIZE                 } from '../modules/local/normalize'
include { BATCHVSBATCH as BATCHVSBATCH_PRE  } from '../modules/local/batchvsbatch'
include { BATCHVSBATCH as BATCHVSBATCH_POST } from '../modules/local/batchvsbatch'
include { OVWT_BATCHWISE            } from '../modules/local/ovwt_batchwise'
include { OVWT_GLOBAL               } from '../modules/local/ovwt_global'
include { AGGREGATE_FEATURE_TYPE as AGGREGATE_FEATURE_TYPE_BATCHWISE } from '../modules/local/aggregate_feature_type'
include { AGGREGATE_FEATURE_TYPE as AGGREGATE_FEATURE_TYPE_GLOBAL    } from '../modules/local/aggregate_feature_type'
include { GENERATE_SPLIT        as GENERATE_SPLIT_BATCHWISE          } from '../modules/local/generate_split'
include { GENERATE_SPLIT        as GENERATE_SPLIT_GLOBAL             } from '../modules/local/generate_split'
include { AGGREGATE_HALF        as AGGREGATE_HALF_BATCHWISE          } from '../modules/local/aggregate_half'
include { AGGREGATE_HALF        as AGGREGATE_HALF_GLOBAL             } from '../modules/local/aggregate_half'
include { CORRELATE_FEATURES    as CORRELATE_FEATURES_BATCHWISE      } from '../modules/local/correlate_features'
include { CORRELATE_FEATURES    as CORRELATE_FEATURES_GLOBAL         } from '../modules/local/correlate_features'
include { BLOCKLIST              as BLOCKLIST_BATCHWISE              } from '../modules/local/blocklist'
include { BLOCKLIST              as BLOCKLIST_GLOBAL                 } from '../modules/local/blocklist'
include { COMBINE_BLOCKLISTS     as COMBINE_BLOCKLISTS_BATCHWISE     } from '../modules/local/combine_blocklists'
include { COMBINE_BLOCKLISTS     as COMBINE_BLOCKLISTS_GLOBAL        } from '../modules/local/combine_blocklists'
include { FINALIZE_FEATURE_SELECT as FINALIZE_FEATURE_SELECT_BATCHWISE } from '../modules/local/finalize_feature_select'
include { FINALIZE_FEATURE_SELECT as FINALIZE_FEATURE_SELECT_GLOBAL    } from '../modules/local/finalize_feature_select'
include { PERMANOVA as PERMANOVA_NORMALIZED     } from '../modules/local/permanova'
include { PERMANOVA as PERMANOVA_BATCH_CORRECTED } from '../modules/local/permanova'
include { BATCH_CORRECT_FIT         } from '../modules/local/batch_correct_fit'
include { BATCH_CORRECT_TRANSFORM   } from '../modules/local/batch_correct_transform'

workflow FisseqPipeline {
    // Validate required parameters (must be inside workflow in DSL2)
    if (params.input_dir == null) {
        error "ERROR: --input_dir is required.\n  Usage: nextflow run fisseq.nf --input_dir /path/to/data"
    }

    // If params.config_dir is set, INPUT generates one input/*.parquet per
    // YAML config file there, merged with any pre-staged files already in
    // <input_dir>/input/. config_files is listed eagerly (not via a Channel)
    // so its basenames can be used synchronously below to dedupe against the
    // pre-staged glob.
    def config_files = []
    if (params.config_dir != null) {
        def configSubdir = file(params.config_dir)
        if (!configSubdir.isDirectory()) {
            error "ERROR: ${params.config_dir} does not exist or is not a directory"
        }
        config_files = configSubdir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
        if (config_files.size() == 0) {
            error "ERROR: No .yaml files found in ${params.config_dir}"
        }
    }
    def config_names = config_files.collect { it.baseName } as Set

    def inputSubdir = file("${params.input_dir}/input")
    def inputParquets = inputSubdir.isDirectory()
        ? (inputSubdir.listFiles()?.findAll { it.name.endsWith('.parquet') } ?: [])
        : []
    if (inputParquets.size() == 0 && config_files.size() == 0) {
        error "ERROR: No .parquet files found in ${params.input_dir}/input and no --config_dir supplied"
    }

    // Resolve input_dir to absolute path so global process scripts can glob published outputs.
    // Relative paths (e.g. ".") break inside Nextflow work directories.
    def input_dir_abs = file(params.input_dir).toAbsolutePath().toString()

    // Pre-staged glob channel, excluding any batch name INPUT will
    // (re-)produce this run. Without this filter, a re-run whose input/
    // already contains a config-derived file from a prior run (published via
    // INPUT's publishDir) would feed that batch into QC_FILTER twice: once
    // from INPUT's live channel output, once from this glob re-matching the
    // file INPUT already published to disk.
    glob_input_ch = Channel.fromPath("${params.input_dir}/input/*.parquet")
        .map { f -> [ f.baseName, f ] }
        .filter { name, f -> !(name in config_names) }

    if (params.config_dir != null) {
        config_ch = Channel.fromList(config_files).map { f -> [ f.baseName, f ] }
        generated_ch = INPUT(config_ch)
        input_ch = glob_input_ch.mix(generated_ch)
    } else {
        input_ch = glob_input_ch
    }

    // Step 1: QC filter (per batch)
    qc_ch = QC_FILTER(input_ch).qc_outputs

    // Step 2: Normalization (per batch)
    // qc_ch carries: (batch_stem, filtered_cells, barcode_counts, variants_per_barcode)
    norm_input_ch = qc_ch.map { stem, fc, bc, vpb -> [ stem, fc ] }
    NORMALIZE(norm_input_ch)
    norm_ch = NORMALIZE.out.normalized  // tuple(batch_stem, normalized_parquet)

    // Single-element signal that fires once all QC_FILTER batches are done.
    // .map preserves the "wait for all batches" dependency while emitting just the path.
    qc_signal = qc_ch.map { stem, fc, bc, vpb -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Single-element signal that fires once all NORMALIZE batches are done.
    global_signal = norm_ch.map { stem, p -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Step 3: Batch-vs-batch — pre batch correction (QC-filtered cells, before normalization)
    BATCHVSBATCH_PRE(qc_signal.map { d -> [d, "${d}/qc_filter/*/filtered_cells.parquet", true, "pre"] })

    // Step 4: Batch-vs-batch — post batch correction (normalized cells)
    BATCHVSBATCH_POST(global_signal.map { d -> [d, "${d}/normalization/cells/*.parquet", false, "post"] })

    // Step 5: OvWT — batchwise
    OVWT_BATCHWISE(norm_ch)

    // Explicit String->Boolean parse: Nextflow CLI overrides (e.g. --global
    // false) arrive as the String "false", which is truthy in Groovy — a
    // bare `if (params.global)` (or `as Boolean`) would run the "disabled"
    // branch anyway. `.toBoolean()` parses "true"/"false" text correctly.
    run_global = params.global.toString().toBoolean()

    // Step 6: OvWT — global (optional, gated on params.global)
    if (run_global) {
        OVWT_GLOBAL(global_signal)
    }

    // Step 7: Feature selection — decomposed bootstrap + per-feature-type pipeline.
    // Stage 1: per-feature-type full aggregation (replaces MultiAggregator).
    // Stage 2a-2d: per-bootstrap pseudo-replicate split -> per-half aggregation
    //   -> correlation -> per-feature-type blocklist (gathered over bootstraps).
    // Stage 3: combine per-feature-type blocklists.
    // Stage 4: join stage-1 aggregates, apply combined blocklist, pycytominer select.
    feature_types_ch = Channel.fromList(params.feature_types)
    // Explicit cast: Nextflow CLI overrides (e.g. --bootstrap 3) arrive as
    // Strings and silently produce a bogus/huge range if left uncoerced in
    // a Groovy IntRange (1..params.bootstrap).
    bootstrap_ch = Channel.of(1..(params.bootstrap as int))

    // --- Batchwise ---

    // Stage 1: full per-feature-type aggregation, one task per (batch, feature_type).
    agg_input_ch = norm_ch
        .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
        .combine(feature_types_ch)
        .map { batch_stem, cells_glob, feature_type ->
            tuple(batch_stem, cells_glob, feature_type, "feature_select_batchwise/${batch_stem}")
        }
    AGGREGATE_FEATURE_TYPE_BATCHWISE(agg_input_ch)
    agg_ch = AGGREGATE_FEATURE_TYPE_BATCHWISE.out  // (batch_stem, feature_type, agg_file)

    // Stage 2a: one 50/50 split per (batch, bootstrap replicate).
    split_input_ch = norm_ch
        .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
        .combine(bootstrap_ch)
        .map { batch_stem, cells_glob, bootstrap_idx ->
            tuple(batch_stem, cells_glob, bootstrap_idx, "feature_select_batchwise/${batch_stem}")
        }
    GENERATE_SPLIT_BATCHWISE(split_input_ch)
    split_ch = GENERATE_SPLIT_BATCHWISE.out  // (batch_stem, bootstrap_idx, half1_file, half2_file)

    // Stage 2b: expand each split into two per-half tuples, cross with feature
    // types, and re-attach the batch's normalized-cells file via
    // .combine(norm_ch, by: 0) (keyed on batch_stem — norm_ch has exactly one
    // entry per batch_stem, so this is a per-batch broadcast, not a fan-out).
    // NOTE: .join() is NOT a broadcast operator — for a many-to-one key
    // relationship like this one it silently keeps only one match per key
    // and drops the rest, which starves all downstream stages. Only use
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

    // Stage 2c: group by (batch_stem, bootstrap_idx, feature_type) — exactly 2
    // per group — pair by half_num (not arrival order) before correlating.
    corr_input_ch = half_agg_ch
        .groupTuple(by: [0, 1, 2])
        // (batch_stem, bootstrap_idx, feature_type, [half_num,half_num], [half_agg_file,half_agg_file])
        .map { batch_stem, bootstrap_idx, feature_type, half_nums, half_files ->
            def pairs = [half_nums, half_files].transpose().sort { it[0] }
            tuple(batch_stem, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1],
                  "feature_select_batchwise/${batch_stem}")
        }
    CORRELATE_FEATURES_BATCHWISE(corr_input_ch)
    corr_ch = CORRELATE_FEATURES_BATCHWISE.out  // (batch_stem, feature_type, bootstrap_idx, correlation_file)

    // Stage 2d: group by (batch_stem, feature_type) — gathers all bootstrap
    // replicates. THE one intentional synchronization point, scoped to this
    // stage only.
    blocklist_input_ch = corr_ch
        .map { batch_stem, feature_type, bootstrap_idx, correlation_file ->
            tuple(batch_stem, feature_type, correlation_file)
        }
        .groupTuple(by: [0, 1])
        // (batch_stem, feature_type, [correlation_file, ...])  (N = params.bootstrap)
        .map { batch_stem, feature_type, correlation_files ->
            tuple(batch_stem, feature_type, correlation_files, "feature_select_batchwise/${batch_stem}")
        }
    BLOCKLIST_BATCHWISE(blocklist_input_ch)
    bl_ch = BLOCKLIST_BATCHWISE.out  // (batch_stem, feature_type, blocklist_file)

    // Stage 3: group by batch_stem — gathers all feature types.
    combine_bl_input_ch = bl_ch
        .map { batch_stem, feature_type, blocklist_file -> tuple(batch_stem, blocklist_file) }
        .groupTuple(by: 0)
        // (batch_stem, [blocklist_file, ...])  (N = params.feature_types.size())
        .map { batch_stem, blocklist_files ->
            tuple(batch_stem, blocklist_files, "feature_select_batchwise/${batch_stem}")
        }
    COMBINE_BLOCKLISTS_BATCHWISE(combine_bl_input_ch)
    combined_bl_ch = COMBINE_BLOCKLISTS_BATCHWISE.out  // (batch_stem, combined_blocklist_file)

    // Stage 4: group stage-1 output by batch_stem (all feature types' full
    // aggregates), join norm_ch (raw cells, for metadata), join stage-3's
    // combined blocklist.
    finalize_input_ch = agg_ch
        .map { batch_stem, feature_type, agg_file -> tuple(batch_stem, agg_file) }
        .groupTuple(by: 0)
        // (batch_stem, [agg_file, ...])  (N = params.feature_types.size())
        .join(norm_ch)
        .join(combined_bl_ch)
        .map { batch_stem, agg_files, normalized_parquet, combined_bl_file ->
            tuple(batch_stem, agg_files, normalized_parquet.toString(), combined_bl_file,
                  "feature_select_batchwise/${batch_stem}")
        }
    FINALIZE_FEATURE_SELECT_BATCHWISE(finalize_input_ch)

    // --- Global (optional, gated on params.global) ---
    // Same shape as batchwise, minus the per-batch dimension: a constant
    // global_key stands in for batch_stem for tuple-shape/grouping purposes
    // only, and the "which cells" glob is derived from global_signal instead
    // of norm_ch (exactly like today's global processes glob published output).
    if (run_global) {
        global_key = "global"

        agg_global_input_ch = global_signal
            .combine(feature_types_ch)
            .map { d, feature_type ->
                tuple(global_key, "${d}/normalization/cells/*.parquet", feature_type, "feature_select_global")
            }
        AGGREGATE_FEATURE_TYPE_GLOBAL(agg_global_input_ch)
        agg_global_ch = AGGREGATE_FEATURE_TYPE_GLOBAL.out  // (global_key, feature_type, agg_file)

        split_global_input_ch = global_signal
            .combine(bootstrap_ch)
            .map { d, bootstrap_idx ->
                tuple(global_key, "${d}/normalization/cells/*.parquet", bootstrap_idx, "feature_select_global")
            }
        GENERATE_SPLIT_GLOBAL(split_global_input_ch)
        split_global_ch = GENERATE_SPLIT_GLOBAL.out  // (global_key, bootstrap_idx, half1_file, half2_file)

        half_global_ch = split_global_ch.flatMap { key, bootstrap_idx, half1, half2 ->
            [
                tuple(key, bootstrap_idx, 1, half1),
                tuple(key, bootstrap_idx, 2, half2),
            ]
        }
        agg_half_global_input_ch = half_global_ch
            .combine(feature_types_ch)
            // (global_key, bootstrap_idx, half_num, index_file, feature_type)
            .combine(global_signal)
            .map { key, bootstrap_idx, half_num, index_file, feature_type, d ->
                tuple(key, bootstrap_idx, half_num, index_file, feature_type,
                      "${d}/normalization/cells/*.parquet", "feature_select_global")
            }
        AGGREGATE_HALF_GLOBAL(agg_half_global_input_ch)
        half_agg_global_ch = AGGREGATE_HALF_GLOBAL.out
        // (global_key, bootstrap_idx, feature_type, half_num, half_agg_file)

        corr_global_input_ch = half_agg_global_ch
            .groupTuple(by: [0, 1, 2])
            .map { key, bootstrap_idx, feature_type, half_nums, half_files ->
                def pairs = [half_nums, half_files].transpose().sort { it[0] }
                tuple(key, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1], "feature_select_global")
            }
        CORRELATE_FEATURES_GLOBAL(corr_global_input_ch)
        corr_global_ch = CORRELATE_FEATURES_GLOBAL.out  // (global_key, feature_type, bootstrap_idx, correlation_file)

        blocklist_global_input_ch = corr_global_ch
            .map { key, feature_type, bootstrap_idx, correlation_file ->
                tuple(key, feature_type, correlation_file)
            }
            .groupTuple(by: [0, 1])
            .map { key, feature_type, correlation_files ->
                tuple(key, feature_type, correlation_files, "feature_select_global")
            }
        BLOCKLIST_GLOBAL(blocklist_global_input_ch)
        bl_global_ch = BLOCKLIST_GLOBAL.out  // (global_key, feature_type, blocklist_file)

        combine_bl_global_input_ch = bl_global_ch
            .map { key, feature_type, blocklist_file -> tuple(key, blocklist_file) }
            .groupTuple(by: 0)
            .map { key, blocklist_files -> tuple(key, blocklist_files, "feature_select_global") }
        COMBINE_BLOCKLISTS_GLOBAL(combine_bl_global_input_ch)
        combined_bl_global_ch = COMBINE_BLOCKLISTS_GLOBAL.out  // (global_key, combined_blocklist_file)

        finalize_global_input_ch = agg_global_ch
            .map { key, feature_type, agg_file -> tuple(key, agg_file) }
            .groupTuple(by: 0)
            .join(combined_bl_global_ch)
            .map { key, agg_files, combined_bl_file ->
                tuple(key, agg_files, "${input_dir_abs}/normalization/cells/*.parquet",
                      combined_bl_file, "feature_select_global")
            }
        FINALIZE_FEATURE_SELECT_GLOBAL(finalize_global_input_ch)
    }

    // Step 9: PERMANOVA — batch-effect assessment (normalized cells)
    PERMANOVA_NORMALIZED(global_signal.map { d -> [d, "${d}/normalization/cells/*.parquet", "permanova"] })

    // New branch: qc_filtering -> batch_correction -> permanova (independent of normalize)
    // Step 1: fit centroid batch correction across all batches (global, waits for all QC_FILTER)
    fit_out = BATCH_CORRECT_FIT(qc_signal).fit_outputs  // tuple(stats_vb, centroids), single emission

    // Step 2: apply batch correction (per batch); .combine() broadcasts the single
    // fit_out pair onto every per-batch tuple from qc_ch.
    bc_transform_input_ch = qc_ch
        .map { stem, fc, bc, vpb -> [ stem, fc ] }
        .combine(fit_out)

    BATCH_CORRECT_TRANSFORM(bc_transform_input_ch)
    bc_ch = BATCH_CORRECT_TRANSFORM.out.corrected  // tuple(batch_stem, corrected_parquet)

    // Single-element signal that fires once all BATCH_CORRECT_TRANSFORM batches are done.
    bc_signal = bc_ch.map { stem, p -> stem }.collect()
        .map { _stems -> input_dir_abs }

    // Step 3: PERMANOVA on batch-corrected cells
    PERMANOVA_BATCH_CORRECTED(bc_signal.map { d -> [d, "${d}/batch_correction/cells/*.parquet", "batch_correction/permanova"] })
}
