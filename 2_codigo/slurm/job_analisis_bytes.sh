#!/bin/bash
#SBATCH --job-name=tesis_analisis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-analisis-%j.out
# Analisis de robustez del clasificador de bytes (Experimento 2c).
# El analisis (a), generalizacion a tipos de archivo nunca vistos, es el critico:
# puede confirmar o matizar el 0,910. Estimado: 40-70 min.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Nucleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar asi:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_analisis_bytes.sh"
    exit 1
fi

python3.11 -u analisis_bytes.py "$DATOS" --por-familia 500

echo "Fin: $(date)"
