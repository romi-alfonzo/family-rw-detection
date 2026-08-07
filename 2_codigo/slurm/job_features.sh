#!/bin/bash
#SBATCH --job-name=tesis_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-features-%j.out
# El cluster asigna solo 2 GB por defecto (DefMemPerNode=2048); hay que pedir
# memoria explícitamente. Máximo permitido: 64 GB (MaxMemPerNode=65536).
# Trabajo B: extrae 275 features estadísticas de los archivos cifrados y entrena.
# Objetivo: demostrar que la indistinguibilidad multiclase persiste con features ricas.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

# El sufijo de las carpetas de NapierOne en el cluster es -small
export NAPIERONE_SUFFIX=small

echo "--- 1/2 Extracción de features ---"
python3.11 -u advanced_features.py "$DATOS" advanced_features.csv || exit 1
ls -lh advanced_features.csv

echo "--- 2/2 Entrenamiento y evaluación ---"
python3.11 -u train_advanced.py advanced_features.csv

echo "Fin: $(date)"
