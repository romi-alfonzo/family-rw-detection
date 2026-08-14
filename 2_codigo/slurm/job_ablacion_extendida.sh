#!/bin/bash
#SBATCH --job-name=tesis_ablacion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-ablacion-%j.out
# Ablacion de ventana extendida (1024, 2048, 4096) + bloque del medio.
# Pedido del tutor en la reunion del 12-08-2026: la curva anterior seguia subiendo
# en 512+512 (0,908) y el grafico no mostraba saturacion.
# Memoria: a 4096+4096 la matriz float32 pesa ~475 MB por copia; de ahi los 64G.
# Estimado: 2-4 h. Si se corta por tiempo, relanzar con --ventanas 64 128 256 512 1024 2048

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Nucleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar asi:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_ablacion_extendida.sh"
    exit 1
fi

python3.11 -u ablacion_ventana_extendida.py "$DATOS" --por-familia 500

echo "Fin: $(date)"
