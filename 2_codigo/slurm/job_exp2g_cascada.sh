#!/bin/bash
#SBATCH --job-name=tesis_2g_cascada
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --nodelist=c2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-exp2g-cascada-%j.out
# Cascada del Exp. 2g (exp2g_examen_cascada.py): decide el contenido si su confianza es >= U (0,5 / 0,7 / 0,9); si
# no, decide el sistema completo. Mismos modelos (job 5818), archivos y nombres que el examen de renombrado (5823).
# Predicciones C1-C4 en el docstring del script. QUÉ MIRAR: el final del .out (tablas, quién decide y C1-C4).
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
MODELO="${MODELO:-resultados_exp2g_modelo_job5818/modelos_exp2g.joblib}"
python3.11 -u exp2g_examen_cascada.py "$MODELO" --max-por-familia 500
echo "Fin: $(date)"
