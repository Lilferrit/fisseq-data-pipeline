nextflow.enable.dsl = 2

// NORMALIZE: output_root namespaces outputs so all batches can share normalization/.
// cells go to normalization/cells/ and normalizers go to normalization/normalizers/
// so that the permanova/ovwt glob "normalization/cells/*.parquet" only hits cell data.
process NORMALIZE {
    errorStrategy 'ignore'
    publishDir "${params.input_dir}/normalization", mode: 'copy', saveAs: { fname ->
        fname.endsWith('.normalizer.parquet') ? "normalizers/${fname}" : "cells/${fname}"
    }

    input:
    tuple val(batch_stem), path(filtered_cells)

    output:
    tuple val(batch_stem), path("${batch_stem}.parquet"), emit: normalized
    path("${batch_stem}.normalizer.parquet"),              emit: normalizer

    script:
    """
    echo "Starting NORMALIZE for ${batch_stem}"
    fisseq-normalize \\
        output_dir=. \\
        output_root=${batch_stem} \\
        input_file=${filtered_cells} \\
        save_normalizer=true
    mv ${batch_stem}.filtered_cells.parquet ${batch_stem}.parquet
    """
}
