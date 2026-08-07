nextflow.enable.dsl = 2

// FisseqPipeline: the default, full end-to-end DAG. Wires together QC_FILTER
// -> NORMALIZE -> ANOVA (normalized) -> ANOVA_BLOCKLIST -> BATCHVSBATCH
// (pre unfiltered / post filtered) -> OVWT (batchwise unfiltered +
// barcode-filtered, global feature-filtered only) -> bootstrap
// feature selection (batchwise/global, gated by params.run_feature_selection)
// -> BATCH_CORRECT_FIT/TRANSFORM -> ANOVA (batch-corrected). WTVWT_BATCHWISE
// (per batch, wildtype cells only, one binary classifier per pair of
// wildtype barcodes) branches off NORMALIZE independently, gated per batch
// by params.run_wtvwt (default true). WTVVARIANTPOOL_BATCHWISE (per batch,
// wildtype-barcode-vs-variant-pool) likewise branches off NORMALIZE
// independently, gated per batch by params.run_wtvvariantpool (default
// false).
// BATCHVSBATCH, OVWT_GLOBAL, ANOVA (both calls), BATCH_CORRECT_FIT/TRANSFORM,
// and the global feature-selection branch all run once per named channel in
// params.global_channels (default null = none run), each scoped to only the
// batches whose YAML `global_channel` key names that channel -- see
// docs/configuration.md#global-channels. This makes the entire ANOVA/
// ANOVA_BLOCKLIST/BATCH_CORRECT_FIT/TRANSFORM/ANOVA_BATCH_CORRECTED chain a
// purely per-channel feature: with no active channels, none of it runs.
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
// Per-batch gating implications (e.g. run_check_barcodes implies
// run_single_cell_scores) are defined once in batchGates() below and baked
// into gates_ch. run_ovwt=false short-circuits this entire chain for that
// batch (single-cell-scores/check-barcodes/barcode-filtered-ovwt all consume
// OVWT_BATCHWISE_UNFILTERED's output), regardless of those params' own settings.
// See AGENTS.md's "Project overview" DAG diagram for the full picture.
include { INPUT                     } from '../modules/local/input'
include { QC_FILTER                 } from '../modules/local/qc_filter'
include { NORMALIZE                 } from '../modules/local/normalize'
include { STAGE_CHANNEL_CELLS as STAGE_CHANNEL_QC   } from '../modules/local/stage_channel'
include { STAGE_CHANNEL_CELLS as STAGE_CHANNEL_NORM } from '../modules/local/stage_channel'
include { BATCHVSBATCH as BATCHVSBATCH_PRE  } from '../modules/local/batchvsbatch'
include { BATCHVSBATCH as BATCHVSBATCH_POST } from '../modules/local/batchvsbatch'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_UNFILTERED       } from '../modules/local/ovwt_batchwise'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_BARCODE_FILTERED } from '../modules/local/ovwt_batchwise'
include { OVWT_GLOBAL               } from '../modules/local/ovwt_global'
include { OVWT_CELLSCORES_BATCHWISE } from '../modules/local/ovwt_cellscores_batchwise'
include { WTVWT_BATCHWISE           } from '../modules/local/wtvwt_batchwise'
include { WTVVARIANTPOOL_BATCHWISE  } from '../modules/local/wtvvariantpool_batchwise'
include { CHECK_BARCODES            } from '../modules/local/check_barcodes'
include { BARCODE_BLOCKLIST         } from '../modules/local/barcode_blocklist'
include { ANOVA_BLOCKLIST           } from '../modules/local/anova_blocklist'
include { AGGREGATE_FEATURE_TYPE as AGGREGATE_FEATURE_TYPE_BATCHWISE } from '../modules/local/aggregate_feature_type'
include { GENERATE_SPLIT        as GENERATE_SPLIT_BATCHWISE          } from '../modules/local/generate_split'
include { AGGREGATE_HALF        as AGGREGATE_HALF_BATCHWISE          } from '../modules/local/aggregate_half'
include { CORRELATE_FEATURES    as CORRELATE_FEATURES_BATCHWISE      } from '../modules/local/correlate_features'
include { BLOCKLIST              as BLOCKLIST_BATCHWISE              } from '../modules/local/blocklist'
include { COMBINE_BLOCKLISTS     as COMBINE_BLOCKLISTS_BATCHWISE     } from '../modules/local/combine_blocklists'
include { FINALIZE_FEATURE_SELECT as FINALIZE_FEATURE_SELECT_BATCHWISE } from '../modules/local/finalize_feature_select'
include { GLOBAL_FEATURE_SELECT     } from '../modules/local/global_feature_select'
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
        wtvwt_max_barcodes                : params.wtvwt_max_barcodes,
        wtvwt_barcode_downsample_mode     : params.wtvwt_barcode_downsample_mode,
        wtvvariantpool_min_cells_per_barcode  : params.wtvvariantpool_min_cells_per_barcode,
        wtvvariantpool_variant_classes        : params.wtvvariantpool_variant_classes,
        wtvvariantpool_downsample_variant_pool: params.wtvvariantpool_downsample_variant_pool,
        feature_select_downsample_wt      : params.feature_select_downsample_wt,
        feature_select_min_correlation    : params.feature_select_min_correlation,
        run_pca                           : BatchParams.asBool(params.run_pca),
        pca_n_components                  : params.pca_n_components,
        run_umap                          : BatchParams.asBool(params.run_umap),
        umap_n_components                 : params.umap_n_components,
        umap_n_neighbors                  : params.umap_n_neighbors,
        umap_metric                       : params.umap_metric,
        umap_min_dist                     : params.umap_min_dist,
        umap_random_state                 : params.umap_random_state,
        barcode_check_min_cells           : params.barcode_check_min_cells,
        barcode_check_alpha               : params.barcode_check_alpha,
        single_cell_scores_split          : params.single_cell_scores_split,
        run_ovwt                          : BatchParams.asBool(params.run_ovwt),
        run_single_cell_scores            : BatchParams.asBool(params.run_single_cell_scores),
        run_check_barcodes                : BatchParams.asBool(params.run_check_barcodes),
        run_barcode_filtered_ovwt         : BatchParams.asBool(params.run_barcode_filtered_ovwt),
        run_feature_selection             : BatchParams.asBool(params.run_feature_selection),
        run_wtvwt                         : BatchParams.asBool(params.run_wtvwt),
        run_wtvvariantpool                : BatchParams.asBool(params.run_wtvvariantpool),
        feature_allowlist_file            : params.feature_allowlist_file,
        feature_blocklist_file            : params.feature_blocklist_file,
        csv_schema_scan_rows              : params.csv_schema_scan_rows,
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
    def config_files = configsDir.listFiles()?.findAll { f -> f.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // Resolve every batch YAML's overrides once, here, at
    // workflow-construction time -- see lib/BatchParams.groovy and
    // docs/configuration.md's "Per-batch parameter overrides" section. Every
    // batch has a YAML (mandatory), so every batch_stem is populated below before
    // any downstream channel closure reads it. Per-batch derived values
    // (e.g. the gating "implies" logic in batchGates()) are computed into
    // NEW maps, never written back into resolvedBatchConfigs.
    def resolvedBatchConfigs = [:]
    config_files.each { f ->
        def batch_stem = f.baseName
        def yamlMap = (new org.yaml.snakeyaml.Yaml().load(f.text) ?: [:]) as Map
        def resolution
        try {
            resolution = BatchParams.resolve(batch_stem, batchParamDefaults, yamlMap)
        } catch (IllegalArgumentException | IllegalStateException e) {
            error "ERROR: ${e.message}"
        }
        resolution.overrides.each { o ->
            log.info "Batch '${o.batch}': overriding ${o.key} (default=${o.defaultValue}) -> ${o.overrideValue}"
        }
        if (!(resolution.resolved.single_cell_scores_split in ["test", "train"])) {
            error "ERROR: batch '${batch_stem}': single_cell_scores_split must be 'test' or 'train', got '${resolution.resolved.single_cell_scores_split}'"
        }
        resolvedBatchConfigs[batch_stem] = resolution.resolved
    }

    // Canonical definition of per-batch gating -- the implications between
    // gate booleans below used to be nested workflow-scope `if`s. Returns a
    // NEW map each call -- never mutates resolvedBatchConfigs. Consumed via
    // gates_ch below, not called directly elsewhere.
    batchGates = { batch_stem ->
        def cfg = resolvedBatchConfigs[batch_stem]
        def runCheckBarcodes = BatchParams.asBool(cfg.run_check_barcodes)
        [
            run_ovwt                 : BatchParams.asBool(cfg.run_ovwt),
            run_check_barcodes       : runCheckBarcodes,
            run_single_cell_scores   : BatchParams.asBool(cfg.run_single_cell_scores) || runCheckBarcodes,
            run_barcode_filtered_ovwt: BatchParams.asBool(cfg.run_barcode_filtered_ovwt) && runCheckBarcodes,
            run_feature_selection    : BatchParams.asBool(cfg.run_feature_selection),
            run_wtvwt                : BatchParams.asBool(cfg.run_wtvwt),
            run_wtvvariantpool       : BatchParams.asBool(cfg.run_wtvvariantpool),
        ]
    }

    // One (batch_stem, gateMap) tuple per batch, computed once via batchGates()
    // above rather than re-invoked per downstream .filter{} -- every gated
    // input channel below .join()s this instead of calling batchGates.call(batch_stem)
    // inline. One entry per batch_stem, same "resolved at workflow-construction
    // time" property as resolvedBatchConfigs.
    // NOTE: `channel.fromList(...)` here is the lowercase channel *factory*
    // (Nextflow's current preferred syntax, replacing the deprecated
    // `Channel.fromList(...)`) -- unrelated to the "never bind a variable
    // named `channel`" rule below; a factory call is not a variable binding.
    gates_ch = channel.fromList(resolvedBatchConfigs.keySet() as List)
        .map { batch_stem -> tuple(batch_stem, batchGates.call(batch_stem)) }

    // Resolve pipeline_dir to absolute path so global process scripts can
    // glob published outputs. Relative paths (e.g. ".") break inside
    // Nextflow work directories.
    def pipeline_dir_abs = file(params.pipeline_dir).toAbsolutePath().toString()

    config_ch = channel.fromList(config_files).map { f ->
        def batch_stem = f.baseName
        def cfg = resolvedBatchConfigs[batch_stem]
        tuple(batch_stem, cfg.input_paths, cfg.feature_allowlist_file, cfg.feature_blocklist_file,
              cfg.csv_schema_scan_rows)
    }
    input_ch = INPUT(config_ch)

    // Step 1: QC filter (per batch)
    qc_input_ch = input_ch.map { batch_stem, f ->
        def cfg = resolvedBatchConfigs[batch_stem]
        tuple(batch_stem, f, cfg.barcode_count_threshold, cfg.variant_barcode_count_threshold,
              cfg.edit_distance_threshold, cfg.qc_n_variants, cfg.qc_variant_downsample_classes,
              cfg.qc_variant_downsample_mode, cfg.qc_downsample_amounts, cfg.qc_downsample_classes,
              cfg.qc_downsample_seed)
    }
    qc_ch = QC_FILTER(qc_input_ch).qc_outputs

    // Step 2: Normalization (per batch)
    // qc_ch carries: (batch_stem, filtered_cells, barcode_counts, variants_per_barcode)
    norm_input_ch = qc_ch.map { batch_stem, fc, _bc, _vpb -> [ batch_stem, fc ] }
    NORMALIZE(norm_input_ch)
    norm_ch = NORMALIZE.out.normalized  // tuple(batch_stem, normalized_parquet)

    // Per-channel fan-out: params.global_channels lists which named channels
    // actually run BATCHVSBATCH/OVWT_GLOBAL/ANOVA/BATCH_CORRECT_FIT+TRANSFORM/
    // the _GLOBAL feature-selection chain -- each gets its own run, scoped to
    // only the batches whose resolved global_channel list names that
    // channel. channels_ch is built via channel.fromList(), the same idiom
    // feature_types_ch/bootstrap_ch use below to fan out N tasks from one
    // process invocation -- NOT a Groovy loop calling a process repeatedly.
    // If params.global_channels is null/[] (the default), channels_ch is
    // empty, so STAGE_CHANNEL_QC/STAGE_CHANNEL_NORM and everything gated on
    // their signals below simply run zero tasks -- no `if` gate needed,
    // consistent with batchGates()'s "filter, don't if" pattern. This makes
    // the entire ANOVA/ANOVA_BLOCKLIST/BATCH_CORRECT_FIT/TRANSFORM/
    // ANOVA_BATCH_CORRECTED chain a purely per-channel feature too: with no
    // active channels, none of it runs.
    def activeChannels = (params.global_channels ?: []) as List<String>
    channels_ch = channel.fromList(activeChannels)

    // NOTE: the per-channel identifier is bound as "chan" in every closure
    // below, never "channel" -- "channel" is a reserved Nextflow binding
    // (lowercase alias for the Channel class) and silently resolves to
    // `nextflow.Channel` itself if reused as a variable name, rather than
    // failing loudly -- see AGENTS.md.

    // Shared by channel_qc_input_ch/channel_norm_input_ch below: scope a
    // (batch_stem, value) channel down to only the batches that name a given
    // channel in their resolved global_channel list, tagging each surviving
    // tuple with `label` (the STAGE_CHANNEL_CELLS subdir to publish under).
    stageChannelInput = { srcCh, label ->
        srcCh.combine(channels_ch)
            .filter { batch_stem, _v, chan -> chan in (resolvedBatchConfigs[batch_stem].global_channel ?: []) }
            .map { batch_stem, v, chan -> tuple(chan, batch_stem, v, label) }
    }
    channel_qc_input_ch = stageChannelInput.call(qc_ch.map { batch_stem, fc, _bc, _vpb -> tuple(batch_stem, fc) }, 'qc_filter_cells')
    STAGE_CHANNEL_QC(channel_qc_input_ch)

    channel_norm_input_ch = stageChannelInput.call(norm_ch, 'normalization_cells')
    STAGE_CHANNEL_NORM(channel_norm_input_ch)

    // Per-channel "wait for all this channel's batches" signal --
    // groupTuple() buffers until the upstream channel closes, giving the
    // same wait-for-everything property qc_signal/global_signal used to get
    // from .collect(), just keyed per channel instead of flattened to one
    // signal. Shared by every STAGE_CHANNEL_*/BATCH_CORRECT_TRANSFORM-derived
    // signal below (channel_qc_signal_ch, channel_norm_signal_ch,
    // channel_bc_signal_ch) -- all consume a (chan, batch_stem, value) source.
    perChannelSignal = { srcCh, subdir ->
        srcCh.map { chan, batch_stem, _f -> tuple(chan, batch_stem) }
            .groupTuple()
            .map { chan, _batch_stems -> tuple(chan, "${pipeline_dir_abs}/global/${chan}/${subdir}") }
    }
    channel_qc_signal_ch = perChannelSignal.call(STAGE_CHANNEL_QC.out, 'qc_filter_cells')
    channel_norm_signal_ch = perChannelSignal.call(STAGE_CHANNEL_NORM.out, 'normalization_cells')

    // Step 2b: WTVWT — batchwise, wildtype-only pairwise barcode classification.
    // Restricted to wildtype cells; trains one binary classifier per pair of
    // wildtype barcodes. Per-batch gated on run_wtvwt (default true), like
    // run_ovwt/run_feature_selection. Independent of the ANOVA/OvWT/
    // feature-selection chains below, so it only needs norm_ch.
    wtvwt_input_ch = norm_ch.join(gates_ch)
        .filter { _batch_stem, _p, gates -> gates.run_wtvwt }
        .map { batch_stem, p, _gates -> tuple(batch_stem, p, resolvedBatchConfigs[batch_stem].wtvwt_min_cells_per_barcode,
                                 resolvedBatchConfigs[batch_stem].wtvwt_max_barcodes,
                                 resolvedBatchConfigs[batch_stem].wtvwt_barcode_downsample_mode) }
    WTVWT_BATCHWISE(wtvwt_input_ch)

    // Step 2c: WTVVARIANTPOOL — batchwise, wildtype-barcode-vs-variant-pool
    // classification. Pools non-wildtype cells whose classified variant class
    // is in wtvvariantpool_variant_classes, then trains one binary classifier
    // per surviving wildtype barcode vs. that pool. Per-batch gated on
    // run_wtvvariantpool (default FALSE, unlike run_wtvwt's default true).
    // Independent of the ANOVA/OvWT/feature-selection chains and of
    // WTVWT_BATCHWISE itself, so it only needs norm_ch.
    wtvvariantpool_input_ch = norm_ch.join(gates_ch)
        .filter { _batch_stem, _p, gates -> gates.run_wtvvariantpool }
        .map { batch_stem, p, _gates -> tuple(batch_stem, p, resolvedBatchConfigs[batch_stem].wtvvariantpool_min_cells_per_barcode,
                                 resolvedBatchConfigs[batch_stem].wtvvariantpool_variant_classes,
                                 resolvedBatchConfigs[batch_stem].wtvvariantpool_downsample_variant_pool) }
    WTVVARIANTPOOL_BATCHWISE(wtvvariantpool_input_ch)

    // ANOVA (normalized) — once per active global channel, scoped to that
    // channel's normalized cells (channel_norm_signal_ch). Its output feeds
    // ANOVA_BLOCKLIST below.
    ANOVA_NORMALIZED(channel_norm_signal_ch.map { chan, d -> tuple(chan, "${d}/*.parquet", "global/${chan}/anova") })

    // ANOVA_BLOCKLIST — derives a feature block-list from ANOVA_NORMALIZED's
    // p-values, once per active global channel. BATCHVSBATCH_POST and
    // OVWT_GLOBAL each join back to their own channel's blocklist below.
    ANOVA_BLOCKLIST(ANOVA_NORMALIZED.out.map { chan, f -> tuple(chan, f, "global/${chan}/anova_blocklist") })
    anova_blocklist_ch = ANOVA_BLOCKLIST.out  // (chan, blocklist_file)

    // Step 3: Batch-vs-batch — pre batch correction (QC-filtered cells, before
    // normalization), once per active global channel. Every STAGE_CHANNEL_CELLS
    // output is flattened to <batch_stem>.parquet regardless of source (see
    // modules/local/stage_channel.nf), so use_parent_name=false uniformly here
    // (unlike the old whole-pipeline glob, which needed
    // qc_filter/*/filtered_cells.parquet's parent-dir naming since every
    // batch shared the same filename). Unfiltered: no dependency on
    // ANOVA_BLOCKLIST, to preserve early/parallel scheduling.
    BATCHVSBATCH_PRE(
        channel_qc_signal_ch.map { chan, d -> [d, "${d}/*.parquet", false, "global/${chan}/batchvsbatch/pre", null] }
    )

    // Step 4: Batch-vs-batch — post batch correction (normalized cells), once
    // per active global channel. Filtered against that channel's own
    // ANOVA_BLOCKLIST output. .join(), not .combine(): both
    // channel_norm_signal_ch and anova_blocklist_ch are already collapsed to
    // exactly one entry per channel, so .join() pairs each channel with its
    // own blocklist rather than cross-producting across channels.
    BATCHVSBATCH_POST(
        channel_norm_signal_ch.join(anova_blocklist_ch)
            .map { chan, d, bl -> [d, "${d}/*.parquet", false, "global/${chan}/batchvsbatch/post", bl] }
    )

    // Step 5: OvWT — batchwise, unfiltered (barcode-filtered is wired below,
    // after CHECK_BARCODES/BARCODE_BLOCKLIST are available). No dependency
    // on ANOVA_BLOCKLIST, per-batch gated on run_ovwt (default true,
    // preserving the old "always runs" behavior). Since score_source_ch/
    // CHECK_BARCODES/BARCODE_BLOCKLIST/OVWT_BATCHWISE_BARCODE_FILTERED below
    // all consume this process's *output*, run_ovwt=false for a batch means
    // that batch emits nothing here and therefore nothing downstream through
    // that whole chain either -- regardless of that batch's own
    // run_single_cell_scores/run_check_barcodes/run_barcode_filtered_ovwt
    // settings. This falls out automatically from filtering the input
    // channel (no separate "implies" logic needed, unlike the
    // run_check_barcodes/run_single_cell_scores pair below).
    ovwt_unfiltered_input_ch = norm_ch.join(gates_ch)
        .filter { _batch_stem, _p, gates -> gates.run_ovwt }
        .map { batch_stem, p, _gates ->
            def cfg = resolvedBatchConfigs[batch_stem]
            tuple(batch_stem, p, null, null, "ovwt_batchwise", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
                  cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
        }
    OVWT_BATCHWISE_UNFILTERED(ovwt_unfiltered_input_ch)

    // Step 5b: single-cell scores (gated on run_single_cell_scores) ->
    // per-batch barcode-outlier check (gated on run_check_barcodes) ->
    // per-batch barcode block-list (gated on run_barcode_filtered_ovwt) ->
    // OVWT_BATCHWISE_BARCODE_FILTERED. Each stage's *input* is the *previous*
    // stage's *output*, so batchGates()'s gating implications (definition
    // above) fall out naturally without re-deriving them at each filter --
    // see nextflow.config's comments for these params.
    score_source_ch = OVWT_BATCHWISE_UNFILTERED.out.join(gates_ch)
        .filter { _batch_stem, _res, _mdl, _test_idx, _train_idx, gates -> gates.run_single_cell_scores }
        .map { batch_stem, _res, mdl, test_idx, train_idx, _gates ->
            def split = resolvedBatchConfigs[batch_stem].single_cell_scores_split
            tuple(batch_stem, (split == "test") ? test_idx : train_idx, mdl)
        }
    OVWT_CELLSCORES_BATCHWISE(score_source_ch)

    check_barcodes_input_ch = OVWT_CELLSCORES_BATCHWISE.out.join(gates_ch)
        .filter { _batch_stem, _scores, gates -> gates.run_check_barcodes }
        .map { batch_stem, scores, _gates ->
            def cfg = resolvedBatchConfigs[batch_stem]
            tuple(batch_stem, scores, cfg.barcode_check_min_cells, cfg.barcode_check_alpha)
        }
    CHECK_BARCODES(check_barcodes_input_ch)

    // Per-batch, unlike ANOVA_BLOCKLIST (per-channel) -- consumes CHECK_BARCODES'
    // per-batch (batch_stem, results_file) tuple directly.
    barcode_blocklist_input_ch = CHECK_BARCODES.out.join(gates_ch)
        .filter { _batch_stem, _res, gates -> gates.run_barcode_filtered_ovwt }
        .map { batch_stem, res, _gates -> tuple(batch_stem, res, resolvedBatchConfigs[batch_stem].barcode_blocklist_pvalue_threshold) }
    BARCODE_BLOCKLIST(barcode_blocklist_input_ch)
    barcode_blocklist_ch = BARCODE_BLOCKLIST.out  // (batch_stem, barcode_blocklist_file)

    // .join(), not .combine(): both norm_ch and barcode_blocklist_ch already
    // carry exactly one entry per batch_stem that made it through
    // CHECK_BARCODES -- .combine() would be a global broadcast, which is
    // wrong here since the blocklist is per-batch, not global.
    OVWT_BATCHWISE_BARCODE_FILTERED(
        norm_ch.join(barcode_blocklist_ch)
            .map { batch_stem, p, bl ->
                def cfg = resolvedBatchConfigs[batch_stem]
                tuple(batch_stem, p, null, bl, "ovwt_batchwise_barcode_filtered", cfg.ovwt_min_cells, cfg.ovwt_downsample_wt,
                      cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
            }
    )

    // Step 6: OvWT — global, once per active global channel. Always filtered
    // against that channel's own ANOVA_BLOCKLIST -- there is no unfiltered
    // global run.
    OVWT_GLOBAL(
        channel_norm_signal_ch.join(anova_blocklist_ch)
            .map { chan, d, bl -> tuple("${d}/*.parquet", bl, "global/${chan}/ovwt_global") }
    )

    // Step 7: Feature selection — decomposed bootstrap + per-feature-type pipeline.
    // Stage 1: per-feature-type full aggregation (replaces MultiAggregator).
    // Stage 2a-2d: per-bootstrap pseudo-replicate split -> per-half aggregation
    //   -> correlation -> per-feature-type blocklist (gathered over bootstraps).
    // Stage 3: combine per-feature-type blocklists.
    // Stage 4: join stage-1 aggregates, apply combined blocklist, pycytominer select.
    // The batchwise portion is per-batch gated on that batch's resolved
    // run_feature_selection (via norm_ch_feature_selected below); the global
    // sub-branch runs once per active global channel, gated on
    // params.run_feature_selection. feature_select_types/
    // feature_select_bootstrap_reps are pipeline-wide-only -- they determine
    // shared fan-out cardinality, not a per-batch scalar -- so
    // feature_types_ch/bootstrap_ch are built unconditionally, outside any gate.
    feature_types_ch = channel.fromList(params.feature_select_types)
    // Explicit cast: Nextflow CLI overrides (e.g. --feature_select_bootstrap_reps 3)
    // arrive as Strings and silently produce a bogus/huge range if left
    // uncoerced in a Groovy IntRange (1..params.feature_select_bootstrap_reps).
    bootstrap_ch = channel.of(1..(params.feature_select_bootstrap_reps as int))

    // --- Batchwise --- (per-batch gated on run_feature_selection; norm_ch is
    // filtered once, independently, here -- every downstream groupTuple/
    // .join() stage automatically only sees the surviving batch keys.)
    norm_ch_feature_selected = norm_ch.join(gates_ch)
        .filter { _batch_stem, _p, gates -> gates.run_feature_selection }
        .map { batch_stem, p, _gates -> tuple(batch_stem, p) }

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
            def pairs = [half_nums, half_files].transpose().sort { pair -> pair[0] }
            tuple(batch_stem, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1],
                  "feature_select_batchwise/${batch_stem}")
        }
    CORRELATE_FEATURES_BATCHWISE(corr_input_ch)
    corr_ch = CORRELATE_FEATURES_BATCHWISE.out  // (batch_stem, feature_type, bootstrap_idx, correlation_file)

    // Stage 2d: group by (batch_stem, feature_type) — gathers all bootstrap
    // replicates. THE one intentional synchronization point, scoped to this
    // stage only.
    blocklist_input_ch = corr_ch
        .map { batch_stem, feature_type, _bootstrap_idx, correlation_file ->
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
        .map { batch_stem, _feature_type, blocklist_file -> tuple(batch_stem, blocklist_file) }
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
        .map { batch_stem, _feature_type, agg_file -> tuple(batch_stem, agg_file) }
        .groupTuple(by: 0)
        // (batch_stem, [agg_file, ...])  (N = params.feature_select_types.size())
        .join(norm_ch_feature_selected)
        .join(combined_bl_ch)
        .map { batch_stem, agg_files, normalized_parquet, combined_bl_file ->
            def cfg = resolvedBatchConfigs[batch_stem]
            tuple(batch_stem, agg_files, normalized_parquet.toString(), combined_bl_file,
                  "feature_select_batchwise/${batch_stem}",
                  cfg.run_pca, cfg.pca_n_components, cfg.run_umap, cfg.umap_n_components,
                  cfg.umap_n_neighbors, cfg.umap_metric, cfg.umap_min_dist, cfg.umap_random_state)
        }
    FINALIZE_FEATURE_SELECT_BATCHWISE(finalize_input_ch)

    // --- Global (once per active global channel, gated on
    // params.run_feature_selection) ---
    // Unlike the old bootstrap-recompute global chain this replaces, no
    // per-channel cell staging is needed here: GLOBAL_FEATURE_SELECT reuses
    // each member batch's already-computed BATCHWISE feature-selection
    // artifacts (feature_select_batchwise/<batch>/{aggregates,blocklist.parquet})
    // directly off pipeline_dir, looping over batch_stems in Python -- see
    // modules/local/global_feature_select.nf and
    // fisseq_data_pipeline.globalfeatureselect.
    if (BatchParams.asBool(params.run_feature_selection)) {
        // Batch -> channel membership is resolved once, synchronously, in
        // Groovy (same "resolved at workflow-construction time" pattern as
        // resolvedBatchConfigs itself) -- only batches with
        // run_feature_selection enabled are included, since only those have
        // feature_select_batchwise/<batch>/{aggregates,blocklist.parquet} on
        // disk for GLOBAL_FEATURE_SELECT to read.
        def batchesByChannel = activeChannels.collectEntries { chan ->
            [chan, resolvedBatchConfigs.findAll { _batch_stem, cfg ->
                (cfg.global_channel ?: []).contains(chan) && BatchParams.asBool(cfg.run_feature_selection)
            }.keySet() as List]
        }
        batchesByChannel.each { chan, batch_stems ->
            if (batch_stems.isEmpty()) {
                log.warn "Global channel '${chan}' has no member batches with " +
                    "run_feature_selection enabled -- GLOBAL_FEATURE_SELECT will " +
                    "have nothing to read for this channel."
            }
        }

        // Single-element signal that fires once every batch's BATCHWISE
        // feature selection (aggregates + blocklist) has been published --
        // same "collect -> map to pipeline_dir_abs" idiom as before, just
        // gated on combined_bl_ch (the last batchwise feature-select
        // artifact) instead of qc_ch/norm_ch.
        feature_select_ready_signal = combined_bl_ch.map { batch_stem, _bl -> batch_stem }.collect()
            .map { _batch_stems -> pipeline_dir_abs }

        global_fs_input_ch = channels_ch
            .combine(feature_select_ready_signal)
            .map { chan, d ->
                tuple(chan, batchesByChannel[chan], d, "global/${chan}/feature_select",
                      params.global_feature_select_min_batches_ok, params.feature_select_types,
                      BatchParams.asBool(params.run_pca), params.pca_n_components,
                      BatchParams.asBool(params.run_umap), params.umap_n_components,
                      params.umap_n_neighbors, params.umap_metric, params.umap_min_dist,
                      params.umap_random_state)
            }
        GLOBAL_FEATURE_SELECT(global_fs_input_ch)
    }

    // ANOVA (normalized) now runs earlier, right after channel_norm_signal_ch
    // is computed — see above, feeding ANOVA_BLOCKLIST.

    // New branch: qc_filtering -> batch_correction -> anova (independent of
    // normalize), once per active global channel.
    // Step 1: fit centroid batch correction, once per channel, scoped to
    // that channel's own QC-filtered batches (channel_qc_signal_ch).
    fit_out = BATCH_CORRECT_FIT(
        channel_qc_signal_ch.map { chan, d -> tuple(chan, "${d}/*.parquet", "global/${chan}/batch_correction/fit") }
    ).fit_outputs  // (chan, stats_vb, centroids)

    // Step 2: apply batch correction, once per (channel, batch) pair -- a
    // batch belonging to multiple channels is corrected once per channel,
    // independently; a batch belonging to no active channel is skipped
    // entirely. .combine(fit_out, by: 0), not .join(): fit_out has exactly
    // one entry per channel, but the batch side fans out N batches per
    // channel -- a genuine many-to-one relationship, the same shape
    // agg_half_input_ch's .combine(norm_ch_feature_selected, by: 0) above
    // uses, and exactly what the .join()-is-not-a-broadcast-operator note
    // above warns against using .join() for.
    bc_transform_input_ch = channel_qc_input_ch
        .map { chan, batch_stem, fc, _label -> tuple(chan, batch_stem, fc) }
        .combine(fit_out, by: 0)
        .map { chan, batch_stem, fc, stats_vb, centroids ->
            tuple(chan, batch_stem, fc, stats_vb, centroids, "global/${chan}/batch_correction/cells")
        }
    BATCH_CORRECT_TRANSFORM(bc_transform_input_ch)
    bc_ch = BATCH_CORRECT_TRANSFORM.out.corrected  // (chan, batch_stem, corrected_parquet)

    // Per-channel "wait for all this channel's batch-correction tasks"
    // signal -- same perChannelSignal() as channel_qc_signal_ch/
    // channel_norm_signal_ch above, applied to BATCH_CORRECT_TRANSFORM's
    // own output.
    channel_bc_signal_ch = perChannelSignal.call(bc_ch, 'batch_correction/cells')

    // Step 3: ANOVA on batch-corrected cells, once per active global channel.
    ANOVA_BATCH_CORRECTED(
        channel_bc_signal_ch.map { chan, d -> tuple(chan, "${d}/*.parquet", "global/${chan}/batch_correction/anova") }
    )
}
