#!/bin/bash
#SBATCH --job-name=tesis_2g_renom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodelist=c2
#SBATCH --time=04:00:00
#SBATCH --output=slurm-exp2g-renombrado-%j.out
# Examen de renombrado del Exp. 2g (exp2g_examen_renombrado.py): el modelo del job 5818 sobre los archivos de
# NapierOne-small que no usó para entrenar, con cuatro nombres (original, sin la extensión agregada, extensión
# aleatoria, extensión de otra familia). Predicciones en el docstring del script.
# QUÉ MIRAR: el final del .out (tabla, acierto por familia y P1-P5).
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
MODELO="${MODELO:-resultados_exp2g_modelo_job5818/modelos_exp2g.joblib}"
python3.11 -u exp2g_examen_renombrado.py "$MODELO" --max-por-familia 500
echo "Fin: $(date)"
