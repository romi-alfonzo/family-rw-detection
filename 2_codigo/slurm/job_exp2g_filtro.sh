#!/bin/bash
#SBATCH --job-name=tesis_2g_filtro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodelist=c2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-exp2g-filtro-%j.out
# Filtro del Exp. 2g (exp2g_examen_filtro.py): la respuesta del sistema completo se acepta solo si está entre las k
# familias más probables del contenido (k = 2 y 3); si no, queda la del contenido. Diseñado después de ver la cascada
# (5828). Mismos modelos (5818), archivos y nombres que los jobs 5823 y 5828. Predicciones F1-F5 en el docstring.
# QUÉ MIRAR: la salida entera del .out (cabecera, aceptación, tablas y F1-F5).
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
MODELO="${MODELO:-resultados_exp2g_modelo_job5818/modelos_exp2g.joblib}"
python3.11 -u exp2g_examen_filtro.py "$MODELO" --max-por-familia 500
echo "Fin: $(date)"
