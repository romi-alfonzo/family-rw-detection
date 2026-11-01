#!/bin/bash
#SBATCH --job-name=tesis_2g_modelo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-exp2g-modelo-%j.out
# Entrena y guarda los dos modelos del Exp. 2g (contenido y completo) con 500 archivos por familia, semilla 0,
# para el examen externo en local (examen_archivos.py). No hay validación cruzada: solo dos ajustes, unos minutos.
# QUÉ BAJAR: resultados_exp2g_modelo_job<ID>/modelos_exp2g.joblib (decenas de MB) a 4_resultados/ (WinSCP).
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
DATOS="${DATOS:-/scratch/ralfonzo/Napierone-small}"
python3.11 -u exp2g_guardar_modelo.py "$DATOS" --por-familia 500 --semilla 0
echo "Fin: $(date)"
