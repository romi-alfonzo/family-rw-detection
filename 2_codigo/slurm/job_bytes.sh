#!/bin/bash
#SBATCH --job-name=tesis_bytes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-bytes-%j.out
# Experimento 2c: clasificación de familias con ML sobre los bytes de cabecera y cola
# de los archivos cifrados, con búsqueda de hiperparámetros anidada.
# NO usa el nombre ni la extensión del archivo: mide qué información hay en el contenido.
# Estimado: 1-2 h con los valores por defecto (200 archivos/familia en la búsqueda,
# 500 en la re-evaluación final).

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --export=ALL,DATOS job_bytes.sh"
    exit 1
fi

python3.11 -u clasificador_bytes.py "$DATOS" --por-familia 200 --por-familia-final 500

echo "Fin: $(date)"
