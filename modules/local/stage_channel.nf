nextflow.enable.dsl = 2

// STAGE_CHANNEL_CELLS: republishes one batch's staged file into a per-channel
// directory, so the *_GLOBAL processes (BATCHVSBATCH, OVWT_GLOBAL, ANOVA,
// BATCH_CORRECT_FIT, the feature-selection _GLOBAL chain) can glob a
// directory containing only that channel's batches, exactly as they glob
// published output today (see AGENTS.md's "global processes glob published
// files, not channel outputs" gotcha) -- just scoped per global channel
// instead of to the whole pipeline_dir.
// Aliased twice in workflows/fisseq.nf (as STAGE_CHANNEL_QC / STAGE_CHANNEL_NORM)
// for the two data sources global processes consume: QC_FILTER's
// filtered_cells (for BATCHVSBATCH_PRE/BATCH_CORRECT_FIT) and NORMALIZE's
// normalized cells (for BATCHVSBATCH_POST/OVWT_GLOBAL/ANOVA/the
// feature-selection global chain).
// Always publishes as a flat <batch_stem>.parquet regardless of source, so
// every channel-scoped global call site can use use_parent_name=false
// uniformly -- see workflows/fisseq.nf.
// The channel identifier is named "chan" below, not "channel" -- "channel"
// is a reserved Nextflow binding (lowercase alias for the Channel class):
// naming a process input/variable "channel" silently resolves to
// `nextflow.Channel` itself in string interpolation instead of the intended
// value (observed as a literal "global/class nextflow.Channel/..." output
// path) rather than failing loudly, so avoid it -- see AGENTS.md.
process STAGE_CHANNEL_CELLS {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.pipeline_dir}/global/${chan}/${source_label}" }, mode: 'copy'

    input:
    // stageAs: NORMALIZE's own output is already named "<batch_stem>.parquet"
    // -- staging it under a fixed, distinct name here (rather than letting
    // it land as "<batch_stem>.parquet" too) avoids colliding with this
    // process's own identically-named output declaration below.
    tuple val(chan), val(batch_stem), path(cells_file, stageAs: 'staged_cells.parquet'), val(source_label)

    output:
    tuple val(chan), val(batch_stem), path("${batch_stem}.parquet")

    script:
    """
    ln -s staged_cells.parquet ${batch_stem}.parquet
    """
}
