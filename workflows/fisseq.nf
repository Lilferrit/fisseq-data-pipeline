nextflow.enable.dsl = 2

// FisseqPipeline: the default, full end-to-end DAG. Wires together QC_FILTER
// -> NORMALIZE -> ANOVA (normalized) -> ANOVA_BLOCKLIST -> BATCHVSBATCH
// (pre unfiltered / post filtered) -> OVWT (batchwise unfiltered + feature-
// filtered + barcode-filtered, global feature-filtered only) -> bootstrap
// feature selection (batchwise/global, gated by params.run_feature_selection)
// -> BATCH_CORRECT_FIT/TRANSFORM -> ANOVA (batch-corrected). WTVWT_BATCHWISE
// (per batch, wildtype cells only, one binary classifier per pair of
// wildtype barcodes) branches off NORMALIZE independently, gated per batch
// by params.run_wtvwt (default true).
// BATCHVSBATCH, OVWT_GLOBAL, and the global feature-selection branch run
// once per named group in params.global_groups (default null = none run),
// each scoped to only the batches whose YAML `global_group` key names that
// group -- see docs/configuration.md#global-groups. ANOVA, ANOVA_BLOCKLIST,
// and BATCH_CORRECT_FIT/TRANSFORM always run regardless, over every batch.
// OVWT_BATCHWISE_UNFILTERED is itself gated per batch by params.run_ovwt
// (default true). It additionally optionally feeds
// OVWT_CELLSCORES_BATCHWISE (per batch, gated by params.run_single_cell_scores,
// scoring params.single_cell_scores_split's "test" or "train" split), which
// in turn optionally feeds CHECK_BARCODES (per batch, gated by
// params.run_check_barcodes -- a per-variant Tukey HSD across barcodes using
// each cell's own-model score as the response variable), which in turn
// optionally feeds BARCODE_BLOCKLIST (per batch, gated by
// params.run_barcode_filtered_ovwt, default true) -> OVWT_BATCHWISE_BARCODE_FILTERED
// (retrains that batch excluding cells with a blocked barcode).
// run_check_barcodes implies run_single_cell_scores, so setting it alone is
// enough; run_barcode_filtered_ovwt does NOT force run_check_barcodes on --
// it only takes effect once run_check_barcodes is independently true, so the
// default (run_check_barcodes=false) output set is unaffected by
// run_barcode_filtered_ovwt's own default. run_ovwt=false short-circuits
// this entire chain for that batch (single-cell-scores/check-barcodes/
// barcode-filtered-ovwt all consume OVWT_BATCHWISE_UNFILTERED's output),
// regardless of those params' own settings.
// See AGENTS.md's "Project overview" DAG diagram for the full picture.
include { INPUT                     } from '../modules/local/input'
include { QC_FILTER                 } from '../modules/local/qc_filter'
include { NORMALIZE                 } from '../modules/local/normalize'
include { STAGE_GROUP_CELLS as STAGE_GROUP_QC   } from '../modules/local/stage_group'
include { STAGE_GROUP_CELLS as STAGE_GROUP_NORM } from '../modules/local/stage_group'
include { BATCHVSBATCH as BATCHVSBATCH_PRE  } from '../modules/local/batchvsbatch'
include { BATCHVSBATCH as BATCHVSBATCH_POST } from '../modules/local/batchvsbatch'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_UNFILTERED       } from '../modules/local/ovwt_batchwise'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_FEATURE_FILTERED } from '../modules/local/ovwt_batchwise'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_BARCODE_FILTERED } from '../modules/local/ovwt_batchwise'
include { OVWT_GLOBAL               } from '../modules/local/ovwt_global'
include { OVWT_CELLSCORES_BATCHWISE } from '../modules/local/ovwt_cellscores_batchwise'
include { WTVWT_BATCHWISE           } from '../modules/local/wtvwt_batchwise'
include { CHECK_BARCODES            } from '../modules/local/check_barcodes'
include { BARCODE_BLOCKLIST         } from '../modules/local/barcode_blocklist'
include { ANOVA_BLOCKLIST           } from '../modules/local/anova_blocklist'
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
include { ANOVA as ANOVA_NORMALIZED     } from '../modules/local/anova'
include { ANOVA as ANOVA_BATCH_CORRECTED } from '../modules/local/anova'
include { BATCH_CORRECT_FIT         } from '../modules/local/batch_correct_fit'
include { BATCH_CORRECT_TRANSFORM   } from '../modules/local/batch_correct_transform'

workflow FisseqPipeline {
    // Validate required parameters (must be inside workflow in DSL2)
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required.\n  Usage: nextflow run fisseq.nf --pipeline_dir /path/to/data"
    }

    // Pipeline-wide defaults for every batch-overridable key (see
    // lib/BatchParams.groovy), pre-coerced exactly the way the rest of this
    // workflow already coerces params.X (CLI overrides like --run_ovwt
    // false arrive as the Groovy-truthy String "false") so a batch YAML's
    // native-typed value compares correctly against an equivalent
    // CLI-supplied default -- see BatchParams.resolve()'s doc comment.
    def batchParamDefaults = [
        barcode_count_threshold           : params.barcode_count_threshold,
        variant_barcode_count_threshold   : params.variant_barcode_count_threshold,
        edit_distance_threshold           : params.edit_distance_threshold,
        qc_n_variants                     : params.qc_n_variants,
        qc_variant_downsample_classes     : params.qc_variant_downsample_classes,
        qc_variant_downsample_mode        : params.qc_variant_downsample_mode,
        qc_downsample_amounts             : params.qc_downsample_amounts,
        qc_downsample_classes             : params.qc_downsample_classes,
        qc_downsample_seed                : params.qc_downsample_seed,
        barcode_blocklist_pvalue_threshold: params.barcode_blocklist_pvalue_threshold,
        ovwt_min_cells                    : params.ovwt_min_cells,
        ovwt_downsample_wt                : params.ovwt_downsample_wt,
        max_cells_per_barcode_wt          : params.max_cells_per_barcode_wt,
        max_cells_per_barcode_variant     : params.max_cells_per_barcode_variant,
        wtvwt_min_cells_per_barcode       : params.wtvwt_min_cells_per_barcode,
        feature_select_downsample_wt      : params.feature_select_downsample_wt,
        feature_select_min_correlation    : params.feature_select_min_correlation,
        barcode_check_min_cells           : params.barcode_check_min_cells,
        barcode_check_alpha               : params.barcode_check_alpha,
        single_cell_scores_split          : params.single_cell_scores_split,
        run_ovwt                          : params.run_ovwt.toString().toBoolean(),
        run_feature_filtered_ovwt         : params.run_feature_filtered_ovwt.toString().toBoolean(),
        run_single_cell_scores            : params.run_single_cell_scores.toString().toBoolean(),
        run_check_barcodes                : params.run_check_barcodes.toString().toBoolean(),
        run_barcode_filtered_ovwt         : params.run_barcode_filtered_ovwt.toString().toBoolean(),
        run_feature_selection             : params.run_feature_selection.toString().toBoolean(),
        run_wtvwt                         : params.run_wtvwt.toString().toBoolean(),
        feature_allowlist_file            : params.feature_allowlist_file,
        feature_blocklist_file            : params.feature_blocklist_file,
    ]
    if (!(batchParamDefaults.single_cell_scores_split in ["test", "train"])) {
        error "ERROR: --single_cell_scores_split must be 'test' or 'train', got '${batchParamDefaults.single_cell_scores_split}'"
    }

    // INPUT generates one input/*.parquet per YAML config file in
    // <pipeline_dir>/configs/ -- mandatory, every batch must have one.
    // config_files is listed eagerly (not via a Channel) so its basenames
    // can be used synchronously below (batch config resolution needs every
    // batch's YAML parsed before any channel is built).
    def configsDir = file("${params.pipeline_dir}/configs")
    if (!configsDir.isDirectory()) {
        error "ERROR: ${params.pipeline_dir}/configs does not exist or is not a directory"
    }
    def config_files = configsDir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // Resolve every batch YAML's overrides once, here, at
    // workflow-construction time -- see lib/BatchParams.groovy and
    // docs/configuration.md's "Per-batch parameter overrides" section. Every
    // batch has a YAML (mandatory), so every stem is populated below before
    // any downstream channel closure reads it. Per-batch derived values
    // (e.g. the gating "implies" logic in batchGates()) are computed into
    // NEW maps, never written back into resolvedBatchConfigs.
    def resolvedBatchConfigs = [:]
    config_files.each { f ->
        def stem = f.baseName
        def yamlMap = (new org.yaml.snakeyaml.Yaml().load(f.text) ?: [:]) as Map
        def resolution
        try {
            resolution = BatchParams.resolve(stem, batchParamDefaults, yamlMap)
        } catch (IllegalArgumentException | IllegalStateException e) {
            error "ERROR: ${e.message}"
        }
        resolution.overrides.each { o ->
            log.info "Batch '${o.batch}': overriding ${o.key} (default=${o.defaultValue}) -> ${o.overrideValue}"
        }
        if (!(resolution.resolved.single_cell_scores_split in ["test", "train"])) {
            error "ERROR: batch '${stem}': single_cell_scores_split must be 'test' or 'train', got '${resolution.resolved.single_cell_scores_split}'"
        }
        resolvedBatchConfigs[stem] = resolution.resolved
    }

    // Per-batch effective gating booleans, baking in the "run_check_barcodes
    // implies run_single_cell_scores" / "run_barcode_filtered_ovwt only
    // takes effect once run_check_barcodes is true" rules that used to be
    // expressed as nested workflow-scope `if`s. Returns a NEW map each
    // call -- never mutates resolvedBatchConfigs.
    batchGates = { stem ->
        def cfg = resolvedBatchConfigs[stem]
        def runCheckBarcodes = cfg.run_check_barcodes.toString().toBoolean()
        [
            run_ovwt                 : cfg.run_ovwt.toString().toBoolean(),
            run_check_barcodes       : runCheckBarcodes,
            run_single_cell_scores   : cfg.run_single_cell_scores.toString().toBoolean() || runCheckBarcodes,
            run_barcode_filtered_ovwt: cfg.run_barcode_filtered_ovwt.toString().toBoolean() && runCheckBarcodes,
            run_feature_filtered_ovwt: cfg.run_feature_filtered_ovwt.toString().toBoolean(),
            run_feature_selection    : cfg.run_feature_selection.toString().toBoolean(),
            run_wtvwt                : cfg.run_wtvwt.toString().toBoolean(),
        ]
    }

    // Resolve pipeline_dir to absolute path so global process scripts can
    // glob published outputs. Relative paths (e.g. ".") break inside
    // Nextflow work directories.
    def pipeline_dir_abs = file(params.pipeline_dir).toAbsolutePath().toString()

    config_ch = Channel.fromList(config_files).map { f ->
        def stem = f.baseName
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, cfg.input_paths, cfg.feature_allowlist_file, cfg.feature_blocklist_file)
    }
    input_ch = INPUT(config_ch)

    // Step 1: QC filter (per batch)
    qc_input_ch = input_ch.map { stem, f ->
        def cfg = resolvedBatchConfigs[stem]
        tuple(stem, f, cfg.barcode_count_threshold, cfg.variant_barcode_count_threshold,
              cfg.edit_distance_threshold, cfg.qc_n_variants, cfg.qc_variant_downsample_classes,
              cfg.qc_variant_downsample_mode, cfg.qc_downsample_amounts, cfg.qc_downsample_classes,
              cfg.qc_downsample_seed)
    }
    qc_ch = QC_FILTER(qc_input_ch).qc_outputs

    // Step 2: Normalization (per batch)
    // qc_ch carries: (batch_stem, filtered_cells, barcode_counts, variants_per_barcode)
    norm_input_ch = qc_ch.map { stem, fc, bc, vpb -> [ stem, fc ] }
    NORMALIZE(norm_input_ch)
    norm_ch = NORMALIZE.out.normalized  // tuple(batch_stem, normalized_parquet)

    // Single-element signal that fires once all QC_FILTER batches are done.
    // .map preserves the "wait for all batches" dependency while emitting just the path.
    qc_signal = qc_ch.map { stem, fc, bc, vpb -> stem }.collect()
        .map { _stems -> pipeline_dir_abs }

    // Single-element signal that fires once all NORMALIZE batches are done.
    global_signal = norm_ch.map { stem, p -> stem }.collect()
        .map { _stems -> pipeline_dir_abs }

    // Per-group fan-out: params.global_groups lists which named groups
    // actually run BATCHVSBATCH/OVWT_GLOBAL/the _GLOBAL feature-selection
    // chain -- each gets its own run, scoped to only the batches whose
    // resolved global_group list names that group. groups_ch is built via
    // Channel.fromList(), the same idiom feature_types_ch/bootstrap_ch use
    // below to fan out N tasks from one process invocation -- NOT a Groovy
    // loop calling a process repeatedly. If params.global_groups is null/[]
    // (the default), groups_ch is empty, so STAGE_GROUP_QC/STAGE_GROUP_NORM
    // and everything gated on their signals below simply run zero tasks --
    // no `if` gate needed, consistent with batchGates()'s "filter, don't
    // if" pattern.
    def activeGroups = (params.global_groups ?: []) as List<String>
    groups_ch = Channel.fromList(activeGroups)

    group_qc_input_ch = qc_ch.map { stem, fc, bc, vpb -> tuple(stem, fc) }
        .combine(groups_ch)
        .filter { stem, fc, group -> group in (resolvedBatchConfigs[stem].global_group ?: []) }
        .map { stem, fc, group -> tuple(group, stem, fc, 'qc_filter_cells') }
    STAGE_GROUP_QC(group_qc_input_ch)

    group_norm_input_ch = norm_ch.combine(groups_ch)
        .filter { stem, p, group -> group in (resolvedBatchConfigs[stem].global_group ?: []) }
        .map { stem, p, group -> tuple(group, stem, p, 'normalization_cells') }
    STAGE_GROUP_NORM(group_norm_input_ch)

    // Per-group "wait for all this group's batches" signal -- groupTuple()
    // buffers until the upstream channel closes, giving the same
    // wait-for-everything property qc_signal/global_signal get from
    // .collect(), just keyed per group instead of flattened to one signal.
    group_qc_signal_ch = STAGE_GROUP_QC.out
        .map { group, stem, f -> tuple(group, stem) }
        .groupTuple()
        .map { group, stems -> tuple(group, "${pipeline_dir_abs}/global/${group}/qc_filter_cells") }
    group_norm_signal_ch = STAGE_GROUP_NORM.out
        .map { group, stem, f -> tuple(group, stem) }
        .groupTuple()
        .map { group, stems -> tuple(group, "${pipeline_dir_abs}/global/${group}/normalization_cells") }

    // Step 2b: WTVWT — batchwise, wildtype-only pairwise barcode classification.
    // Restricted to wildtype cells; trains one binary classifier per pair of
    // wildtype barcodes. Per-batch gated on run_wtvwt (default true), like
    // run_ovwt/run_feature_selection. Independent of the ANOVA/OvWT/
    // feature-selection chains below, so it only needs norm_ch.
    wtvwt_input_ch = norm_ch
        .filter { stem, p -> batchGates.call(stem).run_wtvwt }
        .map { stem, p -> tuple(stem, p, resolvedBatchConfigs[stem].wtvwt_min_cells_per_barcode) }
    WTVWT_BATCHWISE(wtvwt_input_ch)

    // ANOVA (normalized) — moved up from its previous "Step 9" position so
    // its output channel can feed ANOVA_BLOCKLIST here. Always runs,
    // unconditionally (see below for ANOVA_BLOCKLIST).
    ANOVA_NORMALIZED(global_signal.map { d -> [d, "${d}/normalization/cells/*.parquet", "anova"] })

    // ANOVA_BLOCKLIST — derives a feature block-list from ANOVA_NORMALIZED's
    // p-values. Always runs, not gated by any group, since
    // OVWT_BATCHWISE_FEATURE_FILTERED (when enabled), BATCHVSBATCH_POST,
    // and OVWT_GLOBAL all need it.
    ANOVA_BLOCKLIST(ANOVA_NORMALIZED.out)
    anova_blocklist_ch = ANOVA_BLOCKLIST.out  // single-element path channel

    // Step 3: Batch-vs-batch — pre batch correction (QC-filtered cells, before
    // normalization), once per active global group. Every STAGE_GROUP_CELLS
    // output is flattened to <batch_stem>.parquet regardless of source (see
    // modules/local/stage_group.nf), so use_parent_name=false uniformly here
    // (unlike the old whole-pipeline glob, which needed
    // qc_filter/*/filtered_cells.parquet's parent-dir naming since every
    // batch shared the same filename). Unfiltered: no dependency on
    // ANOVA_BLOCKLIST, to preserve early/parallel scheduling.
    BATCHVSBATCH_PRE(
        group_qc_signal_ch.map { group, d -> [d, "${d}/*.parquet", false, "global/${group}/batchvsbatch/pre", null] }
    )

    // Step 4: Batch-vs-batch — post batch correction (normalized cells), once
    // per active global group. Filtered against ANOVA_BLOCKLIST.
    BATCHVSBATCH_POST(
        group_norm_signal_ch.combine(anova_blocklist_ch)
            .map { group, d, bl -> [d, "${d}/*.parquet", false, "global/${group}/batchvsbatch/post", bl] }
    )

    // Step 5: OvWT — batchwise, unfiltered + feature-filtered (barcode-filtered
    // is wired below, after CHECK_BARCODES/BARCODE_BLOCKLIST are available).
    // Unfiltered: no dependency on ANOVA_BLOCKLIST, per-batch gated on
    // run_ovwt (default true, preserving the old "always runs" behavior).
    // Since score_source_ch/CHECK_BARCODES/BARCODE_BLOCKLIST/
    // OVWT_BATCHWISE_BARCODE_FILTERED below all consume this process's
    // *output*, run_ovwt=false for a batch means that batch emits nothing
    // here and therefore nothing downstream through that whole chain either
    // -- regardless of that batch's own run_single_cell_scores/
    // run_check_barcodes/run_barcode_filtered_ovwt settings. This falls out
    // automatically from filtering the input channel (no separate "implies"
    // logic needed, unlike the run_check_barcodes/run_single_cell_scores
    // pair below).
    ovwt_unfiltered_input_ch = norm_ch
        .filter { stem, p -> batchGates.call(stem).run_ovwt }
        .map { stem, p ->
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, p, null, null, "ovwt_batchwise", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
                  cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
        }
    OVWT_BATCHWISE_UNFILTERED(ovwt_unfiltered_input_ch)
    // Feature-filtered: broadcasts the single ANOVA_BLOCKLIST output onto
    // every per-batch tuple, same .combine() idiom as BATCH_CORRECT_TRANSFORM
    // below. Called unconditionally in Groovy source; batches whose resolved
    // run_feature_filtered_ovwt is false are filtered out of the channel
    // (0 tasks for them, behaviorally identical to the process being
    // "absent" -- see batchGates() above).
    feature_filtered_input_ch = norm_ch.combine(anova_blocklist_ch)
        .filter { stem, p, bl -> batchGates.call(stem).run_feature_filtered_ovwt }
        .map { stem, p, bl ->
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, p, bl, null, "ovwt_batchwise_feature_filtered", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
                  cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
        }
    OVWT_BATCHWISE_FEATURE_FILTERED(feature_filtered_input_ch)

    // Step 5b: single-cell scores (per-batch gated on that batch's resolved
    // run_single_cell_scores) -> per-batch barcode-outlier check (gated on
    // run_check_barcodes) -> per-batch barcode block-list (gated on
    // run_barcode_filtered_ovwt) -> OVWT_BATCHWISE_BARCODE_FILTERED. Each
    // stage's *input* is the *previous* stage's *output*, so the
    // "run_check_barcodes implies run_single_cell_scores" /
    // "run_barcode_filtered_ovwt only takes effect once run_check_barcodes
    // is true" relationships fall out naturally from batchGates() without
    // re-deriving them at each filter -- see nextflow.config's comments for
    // these params and batchGates()'s definition above.
    score_source_ch = OVWT_BATCHWISE_UNFILTERED.out
        .filter { stem, _res, mdl, test_idx, train_idx -> batchGates.call(stem).run_single_cell_scores }
        .map { stem, _res, mdl, test_idx, train_idx ->
            def split = resolvedBatchConfigs[stem].single_cell_scores_split
            tuple(stem, (split == "test") ? test_idx : train_idx, mdl)
        }
    OVWT_CELLSCORES_BATCHWISE(score_source_ch)

    check_barcodes_input_ch = OVWT_CELLSCORES_BATCHWISE.out
        .filter { stem, scores -> batchGates.call(stem).run_check_barcodes }
        .map { stem, scores ->
            def cfg = resolvedBatchConfigs[stem]
            tuple(stem, scores, cfg.barcode_check_min_cells, cfg.barcode_check_alpha)
        }
    CHECK_BARCODES(check_barcodes_input_ch)

    // Per-batch, unlike ANOVA_BLOCKLIST (global) -- consumes CHECK_BARCODES'
    // per-batch (batch_stem, results_file) tuple directly.
    barcode_blocklist_input_ch = CHECK_BARCODES.out
        .filter { stem, res -> batchGates.call(stem).run_barcode_filtered_ovwt }
        .map { stem, res -> tuple(stem, res, resolvedBatchConfigs[stem].barcode_blocklist_pvalue_threshold) }
    BARCODE_BLOCKLIST(barcode_blocklist_input_ch)
    barcode_blocklist_ch = BARCODE_BLOCKLIST.out  // (batch_stem, barcode_blocklist_file)

    // .join(), not .combine(): both norm_ch and barcode_blocklist_ch already
    // carry exactly one entry per batch_stem that made it through
    // CHECK_BARCODES -- .combine() would be a global broadcast, which is
    // wrong here since the blocklist is per-batch, not global.
    OVWT_BATCHWISE_BARCODE_FILTERED(
        norm_ch.join(barcode_blocklist_ch)
            .map { stem, p, bl ->
                def cfg = resolvedBatchConfigs[stem]
                tuple(stem, p, null, bl, "ovwt_batchwise_barcode_filtered", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
                      cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
            }
    )

    // Step 6: OvWT — global, once per active global group. Always filtered
    // against ANOVA_BLOCKLIST -- there is no unfiltered global run.
    OVWT_GLOBAL(
        group_norm_signal_ch.combine(anova_blocklist_ch)
            .map { group, d, bl -> tuple("${d}/*.parquet", bl, "global/${group}/ovwt_global") }
    )

    // Step 7: Feature selection — decomposed bootstrap + per-feature-type pipeline.
    // Stage 1: per-feature-type full aggregation (replaces MultiAggregator).
    // Stage 2a-2d: per-bootstrap pseudo-replicate split -> per-half aggregation
    //   -> correlation -> per-feature-type blocklist (gathered over bootstraps).
    // Stage 3: combine per-feature-type blocklists.
    // Stage 4: join stage-1 aggregates, apply combined blocklist, pycytominer select.
    // The batchwise portion is per-batch gated on that batch's resolved
    // run_feature_selection (via norm_ch_feature_selected below); the global
    // sub-branch runs once per active global group, gated on
    // params.run_feature_selection. feature_select_types/
    // feature_select_bootstrap_reps are pipeline-wide-only -- they determine
    // shared fan-out cardinality, not a per-batch scalar -- so
    // feature_types_ch/bootstrap_ch are built unconditionally, outside any gate.
    feature_types_ch = Channel.fromList(params.feature_select_types)
    // Explicit cast: Nextflow CLI overrides (e.g. --feature_select_bootstrap_reps 3)
    // arrive as Strings and silently produce a bogus/huge range if left
    // uncoerced in a Groovy IntRange (1..params.feature_select_bootstrap_reps).
    bootstrap_ch = Channel.of(1..(params.feature_select_bootstrap_reps as int))

    // --- Batchwise --- (per-batch gated on run_feature_selection; norm_ch is
    // filtered once, independently, here -- every downstream groupTuple/
    // .join() stage automatically only sees the surviving batch keys.)
    norm_ch_feature_selected = norm_ch.filter { batch_stem, _p -> batchGates.call(batch_stem).run_feature_selection }

    // Stage 1: full per-feature-type aggregation, one task per (batch, feature_type).
    agg_input_ch = norm_ch_feature_selected
        .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
        .combine(feature_types_ch)
        .map { batch_stem, cells_glob, feature_type ->
            tuple(batch_stem, cells_glob, feature_type, "feature_select_batchwise/${batch_stem}",
                  resolvedBatchConfigs[batch_stem].feature_select_downsample_wt)
        }
    AGGREGATE_FEATURE_TYPE_BATCHWISE(agg_input_ch)
    agg_ch = AGGREGATE_FEATURE_TYPE_BATCHWISE.out  // (batch_stem, feature_type, agg_file)

    // Stage 2a: one 50/50 split per (batch, bootstrap replicate).
    split_input_ch = norm_ch_feature_selected
        .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
        .combine(bootstrap_ch)
        .map { batch_stem, cells_glob, bootstrap_idx ->
            tuple(batch_stem, cells_glob, bootstrap_idx, "feature_select_batchwise/${batch_stem}")
        }
    GENERATE_SPLIT_BATCHWISE(split_input_ch)
    split_ch = GENERATE_SPLIT_BATCHWISE.out  // (batch_stem, bootstrap_idx, half1_file, half2_file)

    // Stage 2b: expand each split into two per-half tuples, cross with feature
    // types, and re-attach the batch's normalized-cells file via
    // .combine(norm_ch_feature_selected, by: 0) (keyed on batch_stem —
    // norm_ch_feature_selected has exactly one entry per surviving
    // batch_stem, so this is a per-batch broadcast, not a fan-out).
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
        .combine(norm_ch_feature_selected, by: 0)
        // (batch_stem, bootstrap_idx, half_num, index_file, feature_type, normalized_parquet)
        .map { batch_stem, bootstrap_idx, half_num, index_file, feature_type, normalized_parquet ->
            tuple(batch_stem, bootstrap_idx, half_num, index_file, feature_type,
                  normalized_parquet.toString(), "feature_select_batchwise/${batch_stem}",
                  resolvedBatchConfigs[batch_stem].feature_select_downsample_wt)
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
        // (batch_stem, feature_type, [correlation_file, ...])  (N = params.feature_select_bootstrap_reps)
        .map { batch_stem, feature_type, correlation_files ->
            tuple(batch_stem, feature_type, correlation_files, "feature_select_batchwise/${batch_stem}",
                  resolvedBatchConfigs[batch_stem].feature_select_min_correlation)
        }
    BLOCKLIST_BATCHWISE(blocklist_input_ch)
    bl_ch = BLOCKLIST_BATCHWISE.out  // (batch_stem, feature_type, blocklist_file)

    // Stage 3: group by batch_stem — gathers all feature types.
    combine_bl_input_ch = bl_ch
        .map { batch_stem, feature_type, blocklist_file -> tuple(batch_stem, blocklist_file) }
        .groupTuple(by: 0)
        // (batch_stem, [blocklist_file, ...])  (N = params.feature_select_types.size())
        .map { batch_stem, blocklist_files ->
            tuple(batch_stem, blocklist_files, "feature_select_batchwise/${batch_stem}")
        }
    COMBINE_BLOCKLISTS_BATCHWISE(combine_bl_input_ch)
    combined_bl_ch = COMBINE_BLOCKLISTS_BATCHWISE.out  // (batch_stem, combined_blocklist_file)

    // Stage 4: group stage-1 output by batch_stem (all feature types' full
    // aggregates), join norm_ch_feature_selected (raw cells, for metadata),
    // join stage-3's combined blocklist.
    finalize_input_ch = agg_ch
        .map { batch_stem, feature_type, agg_file -> tuple(batch_stem, agg_file) }
        .groupTuple(by: 0)
        // (batch_stem, [agg_file, ...])  (N = params.feature_select_types.size())
        .join(norm_ch_feature_selected)
        .join(combined_bl_ch)
        .map { batch_stem, agg_files, normalized_parquet, combined_bl_file ->
            tuple(batch_stem, agg_files, normalized_parquet.toString(), combined_bl_file,
                  "feature_select_batchwise/${batch_stem}")
        }
    FINALIZE_FEATURE_SELECT_BATCHWISE(finalize_input_ch)

    // --- Global (once per active global group, gated on
    // params.run_feature_selection) ---
    // Same shape as batchwise, minus the per-batch dimension: the real group
    // name stands in for batch_stem for tuple-shape/grouping purposes, and
    // the "which cells" glob is derived from group_norm_signal_ch (the
    // per-group directory STAGE_GROUP_NORM published) instead of norm_ch
    // (exactly like today's global processes glob published output, just
    // scoped per group -- see AGENTS.md's "global processes glob published
    // files" gotcha).
    if (params.run_feature_selection.toString().toBoolean()) {
        agg_global_input_ch = group_norm_signal_ch
            .combine(feature_types_ch)
            .map { group, d, feature_type ->
                tuple(group, "${d}/*.parquet", feature_type, "global/${group}/feature_select",
                      params.feature_select_downsample_wt)
            }
        AGGREGATE_FEATURE_TYPE_GLOBAL(agg_global_input_ch)
        agg_global_ch = AGGREGATE_FEATURE_TYPE_GLOBAL.out  // (group, feature_type, agg_file)

        split_global_input_ch = group_norm_signal_ch
            .combine(bootstrap_ch)
            .map { group, d, bootstrap_idx ->
                tuple(group, "${d}/*.parquet", bootstrap_idx, "global/${group}/feature_select")
            }
        GENERATE_SPLIT_GLOBAL(split_global_input_ch)
        split_global_ch = GENERATE_SPLIT_GLOBAL.out  // (group, bootstrap_idx, half1_file, half2_file)

        half_global_ch = split_global_ch.flatMap { key, bootstrap_idx, half1, half2 ->
            [
                tuple(key, bootstrap_idx, 1, half1),
                tuple(key, bootstrap_idx, 2, half2),
            ]
        }
        agg_half_global_input_ch = half_global_ch
            .combine(feature_types_ch)
            // (group, bootstrap_idx, half_num, index_file, feature_type)
            .combine(group_norm_signal_ch, by: 0)
            // (group, bootstrap_idx, half_num, index_file, feature_type, d)
            .map { key, bootstrap_idx, half_num, index_file, feature_type, d ->
                tuple(key, bootstrap_idx, half_num, index_file, feature_type,
                      "${d}/*.parquet", "global/${key}/feature_select",
                      params.feature_select_downsample_wt)
            }
        AGGREGATE_HALF_GLOBAL(agg_half_global_input_ch)
        half_agg_global_ch = AGGREGATE_HALF_GLOBAL.out
        // (group, bootstrap_idx, feature_type, half_num, half_agg_file)

        corr_global_input_ch = half_agg_global_ch
            .groupTuple(by: [0, 1, 2])
            .map { key, bootstrap_idx, feature_type, half_nums, half_files ->
                def pairs = [half_nums, half_files].transpose().sort { it[0] }
                tuple(key, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1], "global/${key}/feature_select")
            }
        CORRELATE_FEATURES_GLOBAL(corr_global_input_ch)
        corr_global_ch = CORRELATE_FEATURES_GLOBAL.out  // (group, feature_type, bootstrap_idx, correlation_file)

        blocklist_global_input_ch = corr_global_ch
            .map { key, feature_type, bootstrap_idx, correlation_file ->
                tuple(key, feature_type, correlation_file)
            }
            .groupTuple(by: [0, 1])
            .map { key, feature_type, correlation_files ->
                tuple(key, feature_type, correlation_files, "global/${key}/feature_select", params.feature_select_min_correlation)
            }
        BLOCKLIST_GLOBAL(blocklist_global_input_ch)
        bl_global_ch = BLOCKLIST_GLOBAL.out  // (group, feature_type, blocklist_file)

        combine_bl_global_input_ch = bl_global_ch
            .map { key, feature_type, blocklist_file -> tuple(key, blocklist_file) }
            .groupTuple(by: 0)
            .map { key, blocklist_files -> tuple(key, blocklist_files, "global/${key}/feature_select") }
        COMBINE_BLOCKLISTS_GLOBAL(combine_bl_global_input_ch)
        combined_bl_global_ch = COMBINE_BLOCKLISTS_GLOBAL.out  // (group, combined_blocklist_file)

        finalize_global_input_ch = agg_global_ch
            .map { key, feature_type, agg_file -> tuple(key, agg_file) }
            .groupTuple(by: 0)
            .join(group_norm_signal_ch)
            .join(combined_bl_global_ch)
            .map { key, agg_files, d, combined_bl_file ->
                tuple(key, agg_files, "${d}/*.parquet",
                      combined_bl_file, "global/${key}/feature_select")
            }
        FINALIZE_FEATURE_SELECT_GLOBAL(finalize_global_input_ch)
    }

    // ANOVA (normalized) now runs earlier, right after global_signal is
    // computed — see above, feeding ANOVA_BLOCKLIST.

    // New branch: qc_filtering -> batch_correction -> anova (independent of normalize)
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
        .map { _stems -> pipeline_dir_abs }

    // Step 3: ANOVA on batch-corrected cells
    ANOVA_BATCH_CORRECTED(bc_signal.map { d -> [d, "${d}/batch_correction/cells/*.parquet", "batch_correction/anova"] })
}
