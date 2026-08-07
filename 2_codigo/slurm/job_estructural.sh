#!/bin/bash
#SBATCH --job-name=tesis_estructural
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-estructural-%j.out
# Experimento 2b: marcas estructurales (magic bytes / metadatos) por familia.
# Rápido: solo lee 128 bytes por archivo. No es MPI, es un proceso liviano.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos asignados: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

python3.11 -u deteccion_estructural.py "$DATOS" --max-archivos 50

echo "Fin: $(date)"
