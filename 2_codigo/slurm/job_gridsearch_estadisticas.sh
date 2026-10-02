#!/bin/bash
#SBATCH --job-name=tesis_gs_stats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-gsstats-%j.out
# Busqueda de hiperparametros para el Experimento 2 (caracteristicas estadisticas).
# Cierra la unica limitacion de optimizacion declarada en la tesis.
# Reutiliza advanced_features.csv: NO vuelve a extraer caracteristicas.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Nucleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ ! -s advanced_features.csv ]; then
    echo "ERROR: falta advanced_features.csv en $(pwd)."
    exit 1
fi

python3.11 -u gridsearch_estadisticas.py advanced_features.csv

echo "Fin: $(date)"
