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
// false). OVWTLOBO_BATCHWISE (per batch, leave-one-barcode-out OvWT
// generalization testing) likewise branches off NORMALIZE independently,
// gated per batch by params.run_ovwt_lobo (default false, like
// run_wtvvariantpool).
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
include { OVWTLOBO_BATCHWISE        } from '../modules/local/ovwtlobo_batchwise'
include { CHECK_BARCODES            } from '../modules/local/check_barcodes'
include { BARCODE_BLOCKLIST         } from '../modules/local/barcode_blocklist'
include { ANOVA_BLOCKLIST           } from '../modules/local/anova_blocklist'
include { AGGREGATE_FEATURE_TYPE as AGGREGATE_FEATURE_TYPE_BATCHWISE } from '../modules/local/aggregate_feature_type'
include { WT_NULL_AGGREGATE      as WT_NULL_AGGREGATE_BATCHWISE      } from '../modules/local/wt_null_aggregate'
include { WT_NULL_BLOCKLIST      as WT_NULL_BLOCKLIST_BATCHWISE      } from '../modules/local/wt_null_blocklist'
include { PASSTHROUGH_BLOCKLIST  as PASSTHROUGH_BLOCKLIST_BATCHWISE  } from '../modules/local/passthrough_blocklist'
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

    // Fail fast on the removed Fisher-z-correlation reproducibility params
    // (replaced by the WT-null bootstrap gate -- see docs/cli/features.md's
    // migration note) rather than silently ignoring them if still set.
    def removedFeatureSelectParams = [
        feature_select_bootstrap_reps              : 'feature_select_wt_null_bootstraps',
        feature_select_min_correlation              : null,
        feature_select_se_multiplier                : null,
        feature_select_bootstrap_variant_downsample : null,
    ]
    removedFeatureSelectParams.each { oldName, newName ->
        if (params.containsKey(oldName)) {
            def suggestion = newName ? " -- use --${newName} instead" : " and has no replacement (the WT-null bootstrap gate has no equivalent knob)"
            error "ERROR: --${oldName} was removed${suggestion}"
        }
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
        qc_variant_allow_list_file        : params.qc_variant_allow_list_file,
        qc_downsample_amounts             : params.qc_downsample_amounts,
        qc_downsample_classes             : params.qc_downsample_classes,
        qc_downsample_seed                : params.qc_downsample_seed,
        barcode_blocklist_pvalue_threshold: params.barcode_blocklist_pvalue_threshold,
        ovwt_min_cells                    : params.ovwt_min_cells,
        ovwt_downsample_wt                : params.ovwt_downsample_wt,
        ovwt_min_cells_per_barcode        : params.ovwt_min_cells_per_barcode,
        max_cells_per_barcode_wt          : params.max_cells_per_barcode_wt,
        max_cells_per_barcode_variant     : params.max_cells_per_barcode_variant,
        wtvwt_min_cells_per_barcode       : params.wtvwt_min_cells_per_barcode,
        wtvwt_max_barcodes                : params.wtvwt_max_barcodes,
        wtvwt_barcode_downsample_mode     : params.wtvwt_barcode_downsample_mode,
        wtvvariantpool_min_cells_per_barcode  : params.wtvvariantpool_min_cells_per_barcode,
        wtvvariantpool_variant_classes        : params.wtvvariantpool_variant_classes,
        wtvvariantpool_downsample_variant_pool: params.wtvvariantpool_downsample_variant_pool,
        ovwt_lobo_min_cells_holdout       : params.ovwt_lobo_min_cells_holdout,
        ovwt_lobo_min_barcodes_per_variant: params.ovwt_lobo_min_barcodes_per_variant,
        feature_select_downsample_wt      : params.feature_select_downsample_wt,
        feature_select_per_barcode        : BatchParams.asBool(params.feature_select_per_barcode),
        feature_select_barcode_column     : params.feature_select_barcode_column,
        feature_select_wt_null_tukey_multiplier: params.feature_select_wt_null_tukey_multiplier,
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
        run_ovwt_lobo                     : BatchParams.asBool(params.run_ovwt_lobo),
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
            run_ovwt_lobo            : BatchParams.asBool(cfg.run_ovwt_lobo),
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
              cfg.qc_variant_downsample_mode, cfg.qc_variant_allow_list_file, cfg.qc_downsample_amounts,
              cfg.qc_downsample_classes, cfg.qc_downsample_seed)
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

    // Per-channel "wait for all this channel's batches, and hand them
    // downstream as real Nextflow path inputs" collector -- groupTuple()
    // buffers until the upstream channel closes, giving the same
    // wait-for-everything property qc_signal/global_signal used to get from
    // .collect(), just keyed per channel instead of flattened to one signal.
    // Shared by every STAGE_CHANNEL_*/BATCH_CORRECT_TRANSFORM-derived
    // channel below (channel_qc_signal_ch, channel_norm_signal_ch,
    // channel_bc_signal_ch) -- all consume a (chan, batch_stem, path) source
    // and emit (chan, [path, ...]). This used to collapse to a directory
    // *string* (`"${pipeline_dir_abs}/global/${chan}/${subdir}"`) for the
    // global processes below to re-glob from disk -- that discarded the
    // individual files' identity, so Nextflow's -resume cache couldn't tell
    // when the underlying file *set* changed (see AGENTS.md gotcha 6, and
    // the ANOVA/BATCHVSBATCH/BATCH_CORRECT_FIT/OVWT_GLOBAL call sites below,
    // which now receive this as a real `path` input instead of a `val` glob).
    perChannelSignal = { srcCh ->
        srcCh.map { chan, _batch_stem, f -> tuple(chan, f) }
            .groupTuple()
    }
    channel_qc_signal_ch = perChannelSignal.call(STAGE_CHANNEL_QC.out)
    channel_norm_signal_ch = perChannelSignal.call(STAGE_CHANNEL_NORM.out)

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

    // Step 2d: OVWTLOBO — batchwise, leave-one-barcode-out OvWT generalization
    // testing. For each non-wildtype variant, repeatedly holds out one barcode,
    // retrains an OvWT-equivalent classifier on the rest, and scores the
    // held-out barcode. Per-batch gated on run_ovwt_lobo (default FALSE, unlike
    // run_wtvwt's default true -- LOBO trains many more models per batch).
    // Independent of the ANOVA/OvWT/feature-selection chains and of
    // WTVWT_BATCHWISE/WTVVARIANTPOOL_BATCHWISE, so it only needs norm_ch.
    // Reuses ovwt_min_cells/ovwt_downsample_wt/max_cells_per_barcode_wt/
    // max_cells_per_barcode_variant from resolvedBatchConfigs (already
    // resolved above for OVWT_BATCHWISE) rather than duplicating them; both
    // block-list values are passed as null (unfiltered), matching
    // OVWT_BATCHWISE_UNFILTERED's call.
    ovwtlobo_input_ch = norm_ch.join(gates_ch)
        .filter { _batch_stem, _p, gates -> gates.run_ovwt_lobo }
        .map { batch_stem, p, _gates ->
            def cfg = resolvedBatchConfigs[batch_stem]
            tuple(batch_stem, p, null, null, cfg.ovwt_min_cells, cfg.ovwt_lobo_min_cells_holdout,
                  cfg.ovwt_lobo_min_barcodes_per_variant, cfg.ovwt_downsample_wt,
                  cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant)
        }
    OVWTLOBO_BATCHWISE(ovwtlobo_input_ch)

    // ANOVA (normalized) — once per active global channel, scoped to that
    // channel's normalized cells (channel_norm_signal_ch, now a real
    // per-channel path list rather than a directory glob -- see
    // perChannelSignal above). Its output feeds ANOVA_BLOCKLIST below.
    ANOVA_NORMALIZED(channel_norm_signal_ch.map { chan, files -> tuple(chan, files, "global/${chan}/anova") })

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
    // ANOVA_BLOCKLIST, to preserve early/parallel scheduling. cells_files is
    // now channel_qc_signal_ch's real per-channel path list (see
    // perChannelSignal above); pipeline_dir_abs fills the unrelated
    // pipeline_dir slot (unused inside batchvsbatch.nf's script -- see that
    // module's comment).
    BATCHVSBATCH_PRE(
        channel_qc_signal_ch.map { chan, files -> [pipeline_dir_abs, files, false, "global/${chan}/batchvsbatch/pre", null] }
    )

    // Step 4: Batch-vs-batch — post batch correction (normalized cells), once
    // per active global channel. Filtered against that channel's own
    // ANOVA_BLOCKLIST output. .join(), not .combine(): both
    // channel_norm_signal_ch and anova_blocklist_ch are already collapsed to
    // exactly one entry per channel, so .join() pairs each channel with its
    // own blocklist rather than cross-producting across channels.
    BATCHVSBATCH_POST(
        channel_norm_signal_ch.join(anova_blocklist_ch)
            .map { chan, files, bl -> [pipeline_dir_abs, files, false, "global/${chan}/batchvsbatch/post", bl] }
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
                  cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant, cfg.ovwt_min_cells_per_barcode)
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
                      cfg.max_cells_per_barcode_wt, cfg.max_cells_per_barcode_variant, cfg.ovwt_min_cells_per_barcode)
            }
    )

    // Step 6: OvWT — global, once per active global channel. Always filtered
    // against that channel's own ANOVA_BLOCKLIST -- there is no unfiltered
    // global run.
    OVWT_GLOBAL(
        channel_norm_signal_ch.join(anova_blocklist_ch)
            .map { chan, files, bl -> tuple(files, bl, "global/${chan}/ovwt_global") }
    )

    // Step 7: Feature selection — decomposed bootstrap + per-feature-type pipeline.
    // Stage 1: per-feature-type full aggregation (replaces MultiAggregator).
    // Stage 2: per-feature-type reproducibility gate, one of two branches:
    //   - WT-null bootstrap (feature_select_wt_null_types): per bootstrap,
    //     split the control pool into two disjoint halves and compute the
    //     aggregator between them (WT_NULL_AGGREGATE), then gather every
    //     bootstrap and apply an upper Tukey fence (WT_NULL_BLOCKLIST).
    //   - Passthrough (every other configured feature type): no
    //     reproducibility computation, every feature marked ok
    //     (PASSTHROUGH_BLOCKLIST), reusing stage 1's aggregate directly.
    // Stage 3: combine both branches' per-feature-type blocklists.
    // Stage 4: join stage-1 aggregates, apply combined blocklist, pycytominer select.
    // The batchwise portion is per-batch gated on that batch's resolved
    // run_feature_selection (via norm_ch_feature_selected below); the global
    // sub-branch runs once per active global channel, gated on
    // params.run_feature_selection. feature_select_types/
    // feature_select_wt_null_types/feature_select_wt_null_bootstraps are
    // pipeline-wide-only -- they determine shared fan-out cardinality and
    // DAG shape, not a per-batch scalar -- so feature_types_ch/
    // wt_null_types/bootstrap_ch are built unconditionally, outside any gate.
    feature_types_ch = channel.fromList(params.feature_select_types)

    // Validate feature_select_wt_null_types once, here, before it's used to
    // build any channel: must be a strict subset of feature_select_types,
    // and none of its entries may be a summary-statistic aggregator (no
    // reference distribution to compare against, so WT-null is ill-defined).
    def wtNullIneligibleTypes = ["mean", "median", "MAD", "std"] as Set
    params.feature_select_wt_null_types.each { t ->
        if (!(t in params.feature_select_types)) {
            error "ERROR: feature_select_wt_null_types entry '${t}' is not in feature_select_types"
        }
        if (t in wtNullIneligibleTypes) {
            error "ERROR: feature_select_wt_null_types entry '${t}' is a summary-statistic " +
                "aggregator (mean/median/MAD/std) -- WT-null reproducibility is ill-defined " +
                "for it; remove it from feature_select_wt_null_types (it is still feature-" +
                "selected via PASSTHROUGH_BLOCKLIST + pycytominer in FINALIZE_FEATURE_SELECT)"
        }
    }
    def wtNullTypesSet = params.feature_select_wt_null_types as Set
    wt_null_types_ch = channel.fromList(params.feature_select_wt_null_types)

    // Explicit cast: Nextflow CLI overrides (e.g.
    // --feature_select_wt_null_bootstraps 3) arrive as Strings and silently
    // produce a bogus/huge range if left uncoerced in a Groovy IntRange
    // (1..params.feature_select_wt_null_bootstraps).
    bootstrap_ch = channel.of(1..(params.feature_select_wt_null_bootstraps as int))

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
                  resolvedBatchConfigs[batch_stem].feature_select_downsample_wt,
                  resolvedBatchConfigs[batch_stem].feature_select_per_barcode,
                  resolvedBatchConfigs[batch_stem].feature_select_barcode_column)
        }
    AGGREGATE_FEATURE_TYPE_BATCHWISE(agg_input_ch)
    agg_ch = AGGREGATE_FEATURE_TYPE_BATCHWISE.out  // (batch_stem, feature_type, agg_file)

    // Stage 2, WT-null branch: one bootstrap replicate per (batch,
    // wt-null-eligible feature type, bootstrap_idx). Crossing
    // norm_ch_feature_selected with wt_null_types_ch/bootstrap_ch (instead
    // of feature_types_ch) means only the configured WT-null subset ever
    // reaches WT_NULL_AGGREGATE -- no filtering needed downstream.
    wt_null_input_ch = norm_ch_feature_selected
        .map { batch_stem, normalized_parquet -> tuple(batch_stem, normalized_parquet.toString()) }
        .combine(wt_null_types_ch)
        .combine(bootstrap_ch)
        .map { batch_stem, cells_glob, feature_type, bootstrap_idx ->
            tuple(batch_stem, feature_type, bootstrap_idx, cells_glob,
                  "feature_select_batchwise/${batch_stem}",
                  resolvedBatchConfigs[batch_stem].feature_select_downsample_wt,
                  resolvedBatchConfigs[batch_stem].feature_select_per_barcode,
                  resolvedBatchConfigs[batch_stem].feature_select_barcode_column)
        }
    WT_NULL_AGGREGATE_BATCHWISE(wt_null_input_ch)
    wt_null_ch = WT_NULL_AGGREGATE_BATCHWISE.out
    // (batch_stem, feature_type, bootstrap_idx, wt_null_file)

    // Group by (batch_stem, feature_type) — gathers all bootstrap
    // replicates. THE one intentional synchronization point, scoped to this
    // branch only.
    wt_null_bl_input_ch = wt_null_ch
        .map { batch_stem, feature_type, _bootstrap_idx, wt_null_file ->
            tuple(batch_stem, feature_type, wt_null_file)
        }
        .groupTuple(by: [0, 1])
        // (batch_stem, feature_type, [wt_null_file, ...])  (N = params.feature_select_wt_null_bootstraps)
        .map { batch_stem, feature_type, wt_null_files ->
            tuple(batch_stem, feature_type, wt_null_files, "feature_select_batchwise/${batch_stem}",
                  resolvedBatchConfigs[batch_stem].feature_select_wt_null_tukey_multiplier)
        }
    WT_NULL_BLOCKLIST_BATCHWISE(wt_null_bl_input_ch)
    wt_null_bl_ch = WT_NULL_BLOCKLIST_BATCHWISE.out  // (batch_stem, feature_type, blocklist_file)

    // Stage 2, passthrough branch: feature types NOT in
    // feature_select_wt_null_types. Filters the ALREADY-computed stage-1
    // agg_ch (every configured feature type) down to just this subset --
    // no cell-level recomputation, unlike the WT-null branch above.
    passthrough_input_ch = agg_ch
        .filter { batch_stem, feature_type, _agg_file -> !(feature_type in wtNullTypesSet) }
        .map { batch_stem, feature_type, agg_file ->
            tuple(batch_stem, feature_type, agg_file, "feature_select_batchwise/${batch_stem}")
        }
    PASSTHROUGH_BLOCKLIST_BATCHWISE(passthrough_input_ch)
    passthrough_bl_ch = PASSTHROUGH_BLOCKLIST_BATCHWISE.out  // (batch_stem, feature_type, blocklist_file)

    // Merge both branches -- every configured feature type produces exactly
    // one blocklist file, via whichever branch it was routed to above.
    bl_ch = wt_null_bl_ch.mix(passthrough_bl_ch)  // (batch_stem, feature_type, blocklist_file)

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
    // artifacts (agg_ch/combined_bl_ch, produced above by
    // AGGREGATE_FEATURE_TYPE_BATCHWISE/COMBINE_BLOCKLISTS_BATCHWISE) as real
    // Nextflow `path` inputs, scoped per channel the same way
    // channel_qc_input_ch/channel_norm_input_ch scope STAGE_CHANNEL_CELLS
    // above -- not re-derived from pipeline_dir/batch_stems via a Python-side
    // glob (see modules/local/global_feature_select.nf and
    // fisseq_data_pipeline.globalfeatureselect for why the old approach broke
    // -resume cache invalidation).
    if (BatchParams.asBool(params.run_feature_selection)) {
        // Batch -> channel membership, resolved once, synchronously, in
        // Groovy (same "resolved at workflow-construction time" pattern as
        // resolvedBatchConfigs itself) -- kept purely for this diagnostic
        // warning; the actual per-channel file wiring below is driven by
        // agg_ch/combined_bl_ch directly, not by this map, so a channel with
        // no member batches simply yields zero rows there (and
        // GLOBAL_FEATURE_SELECT is never invoked for it) rather than being
        // invoked once and failing under errorStrategy 'ignore' as before.
        def batchesByChannel = activeChannels.collectEntries { chan ->
            [chan, resolvedBatchConfigs.findAll { _batch_stem, cfg ->
                (cfg.global_channel ?: []).contains(chan) && BatchParams.asBool(cfg.run_feature_selection)
            }.keySet() as List]
        }
        batchesByChannel.each { chan, batch_stems ->
            if (batch_stems.isEmpty()) {
                log.warn "Global channel '${chan}' has no member batches with " +
                    "run_feature_selection enabled -- GLOBAL_FEATURE_SELECT will " +
                    "not run for this channel."
            }
        }

        // Scope agg_ch/combined_bl_ch to each active channel's member
        // batches (already all run_feature_selection=true by construction,
        // since both descend from norm_ch_feature_selected -- no need to
        // re-check that gate here), then groupTuple() per channel into
        // parallel (batch_stem, file) lists -- mirrors
        // channel_qc_input_ch/channel_norm_input_ch's stageChannelInput
        // idiom above, adapted for agg_ch's extra feature_type element
        // (dropped: join_feature_type_files joins by file content/schema,
        // not filename, so feature_type identity isn't needed downstream).
        channel_agg_input_ch = agg_ch.combine(channels_ch)
            .filter { batch_stem, _ft, _f, chan -> chan in (resolvedBatchConfigs[batch_stem].global_channel ?: []) }
            .map { batch_stem, _ft, f, chan -> tuple(chan, batch_stem, f) }
            .groupTuple()
            // (chan, [batch_stem, ...], [agg_file, ...])
        channel_bl_input_ch = combined_bl_ch.combine(channels_ch)
            .filter { batch_stem, _bl, chan -> chan in (resolvedBatchConfigs[batch_stem].global_channel ?: []) }
            .map { batch_stem, bl, chan -> tuple(chan, batch_stem, bl) }
            .groupTuple()
            // (chan, [batch_stem, ...], [blocklist_file, ...])

        // .join(), not .combine(): both sides are already collapsed to
        // exactly one row per channel by the groupTuple()s above -- same
        // justification as anova_blocklist_ch's joins elsewhere in this
        // workflow.
        global_fs_input_ch = channel_agg_input_ch.join(channel_bl_input_ch)
            .map { chan, agg_batch_stems, agg_files, bl_batch_stems, bl_files ->
                tuple(chan, agg_batch_stems, agg_files, bl_batch_stems, bl_files,
                      "global/${chan}/feature_select", params.global_feature_select_min_batches_ok,
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
        channel_qc_signal_ch.map { chan, files -> tuple(chan, files, "global/${chan}/batch_correction/fit") }
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
    // collector -- same perChannelSignal() as channel_qc_signal_ch/
    // channel_norm_signal_ch above, applied to BATCH_CORRECT_TRANSFORM's
    // own output.
    channel_bc_signal_ch = perChannelSignal.call(bc_ch)

    // Step 3: ANOVA on batch-corrected cells, once per active global channel.
    ANOVA_BATCH_CORRECTED(
        channel_bc_signal_ch.map { chan, files -> tuple(chan, files, "global/${chan}/batch_correction/anova") }
    )
}
