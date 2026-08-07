#!/bin/bash
#SBATCH --job-name=tesis_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-train-%j.out
# SOLO la etapa de entrenamiento del Trabajo B. NO vuelve a extraer features:
# reutiliza el advanced_features.csv ya generado (esa extracción tomó 5,5 h).
# Se separó del job_features.sh porque el intento anterior (job 3538) completó la
# extracción y murió por OOM recién al entrenar.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ ! -s advanced_features.csv ]; then
    echo "ERROR: no existe o está vacío advanced_features.csv en $(pwd)."
    echo "       Hay que correr primero job_features.sh (tarda ~5,5 h)."
    exit 1
fi
echo "Entrada: $(ls -lh advanced_features.csv | awk '{print $5}') | \
$(wc -l < advanced_features.csv) líneas (1 encabezado + muestras)"

python3.11 -u train_advanced.py advanced_features.csv

echo "Fin: $(date)"
