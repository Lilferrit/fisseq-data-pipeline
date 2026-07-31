nextflow.enable.dsl = 2

// STAGE_GROUP_CELLS: republishes one batch's staged file into a per-group
// directory, so the *_GLOBAL processes (BATCHVSBATCH, OVWT_GLOBAL, the
// feature-selection _GLOBAL chain) can glob a directory containing only that
// group's batches, exactly as they glob published output today (see
// AGENTS.md's "global processes glob published files, not channel outputs"
// gotcha) -- just scoped per group instead of to the whole pipeline_dir.
// Aliased twice in workflows/fisseq.nf (as STAGE_GROUP_QC / STAGE_GROUP_NORM)
// for the two data sources global processes consume: QC_FILTER's
// filtered_cells (for BATCHVSBATCH_PRE) and NORMALIZE's normalized cells
// (for BATCHVSBATCH_POST/OVWT_GLOBAL/the feature-selection global chain).
// Always publishes as a flat <batch_stem>.parquet regardless of source, so
// every group-scoped global call site can use use_parent_name=false
// uniformly -- see workflows/fisseq.nf.
process STAGE_GROUP_CELLS {
    errorStrategy 'ignore'
    label 'process_low'
    publishDir { "${params.pipeline_dir}/global/${group}/${source_label}" }, mode: 'copy'

    input:
    // stageAs: NORMALIZE's own output is already named "<batch_stem>.parquet"
    // -- staging it under a fixed, distinct name here (rather than letting
    // it land as "<batch_stem>.parquet" too) avoids colliding with this
    // process's own identically-named output declaration below.
    tuple val(group), val(batch_stem), path(cells_file, stageAs: 'staged_cells.parquet'), val(source_label)

    output:
    tuple val(group), val(batch_stem), path("${batch_stem}.parquet")

    script:
    """
    ln -s staged_cells.parquet ${batch_stem}.parquet
    """
}
