// lib/BatchParams.groovy — per-batch YAML parameter override resolution.
// See docs/configuration.md's "Per-batch parameter overrides" and "Global
// channels" sections for the full mechanism description and rationale.
//
// This class is pure and side-effect-free (no log.info, no file I/O, no
// dependency on a live Nextflow session or `log` binding) so it can be
// unit-tested directly without a Nextflow runtime. All Nextflow-side-
// effecting concerns (log.info, `error "ERROR: ..."`) are left to the
// calling workflow, which inspects this class's return value — see
// workflows/fisseq.nf / workflows/ovwt.nf for the call site.
class BatchParams {

    // Keys a batch YAML is allowed to override, each with a pipeline-wide
    // nextflow.config default supplied via the `defaults` map passed into
    // resolve(). Every key here must also be present in `defaults` — see
    // resolve()'s defensive IllegalStateException below.
    static final List<String> OVERRIDABLE_KEYS = [
        // QC_FILTER
        'barcode_count_threshold',
        'variant_barcode_count_threshold',
        'edit_distance_threshold',
        'qc_n_variants',
        'qc_variant_downsample_classes',
        'qc_variant_downsample_mode',
        'qc_downsample_amounts',
        'qc_downsample_classes',
        'qc_downsample_seed',
        // BARCODE_BLOCKLIST
        'barcode_blocklist_pvalue_threshold',
        // OVWT_BATCHWISE (shared name with OVWT_GLOBAL, which always uses
        // the plain params.X default at its own call site — see
        // workflows/fisseq.nf)
        'ovwt_min_cells',
        'ovwt_downsample_wt',
        'max_cells_per_barcode_wt',
        'max_cells_per_barcode_variant',
        // WTVWT_BATCHWISE
        'wtvwt_min_cells_per_barcode',
        'wtvwt_max_barcodes',
        'wtvwt_barcode_downsample_mode',
        // WTVVARIANTPOOL_BATCHWISE
        'wtvvariantpool_min_cells_per_barcode',
        'wtvvariantpool_variant_classes',
        'wtvvariantpool_downsample_variant_pool',
        // feature-selection batchwise chain (shared names with the global
        // chain, which always uses the plain params.X default)
        'feature_select_downsample_wt',
        'feature_select_min_correlation',
        // CHECK_BARCODES
        'barcode_check_min_cells',
        'barcode_check_alpha',
        // which OVWT_BATCHWISE split feeds OVWT_CELLSCORES_BATCHWISE
        'single_cell_scores_split',
        // gating booleans with a per-batch-only effect
        'run_ovwt',
        'run_single_cell_scores',
        'run_check_barcodes',
        'run_barcode_filtered_ovwt',
        'run_feature_selection',
        'run_wtvwt',
        'run_wtvvariantpool',
        // INPUT-stage tunables — naturally global, only ever settable
        // per-batch before this refactor; now also plain nextflow.config
        // pipeline-wide defaults (see nextflow.config)
        'feature_allowlist_file',
        'feature_blocklist_file',
        'csv_schema_scan_rows',
    ] as List<String>

    // Keys that exist in nextflow.config but can never be set from a batch
    // YAML: either they gate/consume ALL batches uniformly (global-only
    // consumer, e.g. anova_blocklist_pvalue_threshold) or they're
    // meta/bootstrap params resolved before any batch config exists
    // (pipeline_mode/pipeline_dir). Kept as an explicit list (rather than
    // "anything not in OVERRIDABLE_KEYS") purely so resolve() can give a
    // clearer, more specific error for this case than for a genuine typo —
    // see resolve() below.
    static final List<String> PIPELINE_WIDE_ONLY_KEYS = [
        'pipeline_mode',
        'pipeline_dir',
        'global_channels',
        'feature_select_types',
        'feature_select_bootstrap_reps',
        'global_feature_select_min_batches_ok',
        'anova_blocklist_pvalue_threshold',
        'batchvsbatch_min_cells',
        'batchvsbatch_min_batches',
    ] as List<String>

    // The two batch-YAML-only keys, excluded from the general
    // override-symmetry mechanism (OVERRIDABLE_KEYS/PIPELINE_WIDE_ONLY_KEYS):
    // input_paths (required, no nextflow.config default -- see its handling
    // below) and global_channel (optional, also no nextflow.config default --
    // a batch not naming any channel simply never contributes to a global run).
    static final String INPUT_PATHS_KEY = 'input_paths'
    static final String GLOBAL_CHANNEL_KEY = 'global_channel'

    /**
     * Resolve one batch's fully-merged config.
     *
     * @param batchName human-readable batch identifier (the YAML file's
     *        basename), used only for error/log messages.
     * @param defaults  Map<String,Object> of OVERRIDABLE_KEYS -> that key's
     *        pipeline-wide value. IMPORTANT: build this from the
     *        already-coerced values the workflow uses elsewhere (e.g. the
     *        parsed Boolean for run_check_barcodes, not the raw
     *        possibly-String params.run_check_barcodes) — otherwise a batch
     *        YAML's native-typed value (YAML `true` parses as a Groovy
     *        Boolean) will spuriously compare unequal to a CLI-supplied
     *        String default that means the same thing, producing bogus
     *        "overriding" log noise. See workflows/fisseq.nf's
     *        `batchParamDefaults` construction.
     * @param batchYaml Map as parsed from the batch's YAML file (null-safe:
     *        treated as empty).
     * @return a BatchParamsResolution.
     * @throws IllegalArgumentException on any validation failure (missing
     *         input_paths, pipeline-wide-only key used, unrecognized key).
     * @throws IllegalStateException if OVERRIDABLE_KEYS and `defaults` are
     *         out of sync (a programming error, not a user error).
     */
    static BatchParamsResolution resolve(String batchName, Map<String, Object> defaults, Map<String, Object> batchYaml) {
        if (batchYaml == null) {
            batchYaml = [:]
        }

        // 1. input_paths: required, non-empty, batch-YAML-only.
        def inputPaths = batchYaml.get(INPUT_PATHS_KEY)
        if (!(inputPaths instanceof List) || inputPaths.isEmpty()) {
            throw new IllegalArgumentException(
                "Batch '${batchName}': '${INPUT_PATHS_KEY}' is required and must be a " +
                "non-empty list in the batch YAML (got: ${inputPaths})"
            )
        }

        // 1b. global_channel: optional, batch-YAML-only. A bare string or a
        // list of strings, normalized to a list (empty if omitted). Names
        // which global_channels (see nextflow.config's params.global_channels)
        // this batch contributes to -- a batch naming no channel is simply
        // excluded from every global run.
        def rawChannel = batchYaml.get(GLOBAL_CHANNEL_KEY)
        def globalChannel
        if (rawChannel == null) {
            globalChannel = []
        } else if (rawChannel instanceof String) {
            globalChannel = [rawChannel]
        } else if (rawChannel instanceof List && rawChannel.every { it instanceof String }) {
            globalChannel = rawChannel
        } else {
            throw new IllegalArgumentException(
                "Batch '${batchName}': '${GLOBAL_CHANNEL_KEY}' must be a string or a list " +
                "of strings (got: ${rawChannel})"
            )
        }

        // 2. Validate every other key and build the resolved map + override log.
        def overrides = []
        def resolved = new LinkedHashMap<String, Object>(defaults)
        resolved[INPUT_PATHS_KEY] = inputPaths
        resolved[GLOBAL_CHANNEL_KEY] = globalChannel

        batchYaml.each { key, value ->
            if (key == INPUT_PATHS_KEY || key == GLOBAL_CHANNEL_KEY) {
                return // handled above
            }
            if (key in PIPELINE_WIDE_ONLY_KEYS) {
                throw new IllegalArgumentException(
                    "Batch '${batchName}': '${key}' is a pipeline-wide-only parameter and " +
                    "cannot be set from a batch YAML (it has no per-batch meaning — see " +
                    "docs/nextflow.md's 'Per-batch parameter overrides' section). Set " +
                    "--${key} on the nextflow command line or in nextflow.config instead."
                )
            }
            if (!(key in OVERRIDABLE_KEYS)) {
                throw new IllegalArgumentException(
                    "Batch '${batchName}': unrecognized batch YAML key '${key}' — not a known " +
                    "nextflow.config parameter (check for a typo). See docs/nextflow.md for the " +
                    "full list of batch-overridable keys."
                )
            }
            if (!defaults.containsKey(key)) {
                throw new IllegalStateException(
                    "BatchParams.OVERRIDABLE_KEYS contains '${key}' but the caller's `defaults` " +
                    "map didn't supply a value for it — add it to batchParamDefaults."
                )
            }
            if (value != defaults[key]) {
                overrides << [batch: batchName, key: key, defaultValue: defaults[key], overrideValue: value]
                resolved[key] = value
            }
        }

        return new BatchParamsResolution(resolved: resolved, overrides: overrides)
    }
}

/** Plain data holder returned by BatchParams.resolve() — see its doc comment. */
class BatchParamsResolution {
    Map<String, Object> resolved
    List<Map> overrides
}
