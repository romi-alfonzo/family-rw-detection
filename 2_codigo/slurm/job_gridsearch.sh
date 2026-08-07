#!/bin/bash
#SBATCH --job-name=tesis_gridsearch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-gridsearch-%j.out
# Memoria: el intento anterior (job 3539) murió por OOM. Cada worker paralelo
# mantiene su propia matriz TF-IDF y el coef_ denso del clasificador, así que se
# bajó de 16 a 8 workers, se pidió memoria explícita y se acotó max_features.
# Trabajo A: búsqueda anidada de hiperparámetros del clasificador de notas.
# El más largo (horas). scikit-learn paraleliza la grilla sobre SLURM_CPUS_PER_TASK
# núcleos: el script lee esa variable, así que NO sobresuscribe el nodo.
# Si el cluster está ocupado, bajar --cpus-per-task para que entre antes en cola.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

python3.11 -u gridsearch_notas.py

echo "Fin: $(date)"
