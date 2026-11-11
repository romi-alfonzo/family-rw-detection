#!/bin/bash
#SBATCH --job-name=tesis_2g_confusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodelist=c2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-exp2g-confusion-%j.out
# Matriz de confusión del resultado final del frente de archivos (pedido del tutor, 11-11): el mismo examen del filtro
# (exp2g_examen_filtro.py, mismos modelos 5818, mismos 14.948 archivos no vistos y mismos nombres que el job del
# filtro), que ahora guarda, con el nombre original, la matriz de confusión de cada modelo y las predicciones por
# archivo en resultados_exp2g_filtro_job<NRO>/.
# QUÉ MIRAR: que el filtro k=2 con el nombre original vuelva a dar 0,979 de macro-F1, y la línea «matriz de confusión
# del filtro k=2 ... diagonal / total = 0,9786».
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
MODELO="${MODELO:-resultados_exp2g_modelo_job5818/modelos_exp2g.joblib}"
python3.11 -u exp2g_examen_filtro.py "$MODELO" --max-por-familia 500
echo "Fin: $(date)"
