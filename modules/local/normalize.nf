nextflow.enable.dsl = 2

// NORMALIZE: output_root namespaces outputs so all batches can share normalization/.
// cells go to normalization/cells/ and normalizers go to normalization/normalizers/
// so that the anova/ovwt glob "normalization/cells/*.parquet" only hits cell data.
process NORMALIZE {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir "${params.pipeline_dir}/normalization", mode: 'copy', saveAs: { fname ->
        fname.endsWith('.normalizer.parquet') ? "normalizers/${fname}" : "cells/${fname}"
    }

    input:
    tuple val(batch_stem), path(filtered_cells)

    output:
    tuple val(batch_stem), path("${batch_stem}.parquet"), emit: normalized
    path("${batch_stem}.normalizer.parquet"),              emit: normalizer

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting NORMALIZE for ${batch_stem}"
    python -m fisseq_data_pipeline.normalize \\
        output_dir=. \\
        output_root=${batch_stem} \\
        input_file=${filtered_cells} \\
        save_normalizer=true
    mv ${batch_stem}.filtered_cells.parquet ${batch_stem}.parquet
    """
}
