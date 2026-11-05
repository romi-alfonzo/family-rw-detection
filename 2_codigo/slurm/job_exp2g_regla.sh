#!/bin/bash
#SBATCH --job-name=tesis_2g_regla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodelist=c2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-exp2g-regla-%j.out
# Regla de ausencia del Exp. 2i (exp2g_examen_regla.py): si el archivo no trae una extensión agregada, decide solo el
# contenido; si la trae, el filtro k=2 (o el sistema completo). Diseñada después de ver el filtro (5830). Mismos
# modelos (5818), archivos y nombres que los jobs 5823, 5828 y 5830. Predicciones R1-R5 en el docstring.
# QUÉ MIRAR: la salida entera del .out (cabecera, ruta, tablas, F1 de las familias vigiladas y R1-R5).
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
MODELO="${MODELO:-resultados_exp2g_modelo_job5818/modelos_exp2g.joblib}"
python3.11 -u exp2g_examen_regla.py "$MODELO" --max-por-familia 500
echo "Fin: $(date)"
