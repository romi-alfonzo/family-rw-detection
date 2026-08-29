#!/bin/bash
#SBATCH --job-name=tesis_exp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-exp2d-%j.out
# Experimento 2d: aporte del NOMBRE y de la EXTENSIÓN sobre los bytes.
# Elemento de acción 2 del tutor (reunión del 12-08-2026).
#
# Tres columnas sobre la MISMA partición, para que sean comparables:
#   (1) solo bytes ................... la configuración canónica del Exp. 2c
#   (2) + forma del nombre ........... defendible: longitud, composición, entropía
#   (3) + extensión literal .......... COTA SUPERIOR declarada, no un método
#
# Antes de entrenar, el script MIDE si la extensión determina la etiqueta en este
# conjunto (cuenta extensiones por familia y evalúa una tabla de consulta). Si esa
# tabla ya acierta casi todo, la columna (3) se reporta como memorización de un
# diccionario y no como capacidad de identificar familias.
#
# No repite la búsqueda de hiperparámetros: usa los ya elegidos por el Exp. 2c, porque
# lo que se mide es el aporte de las características y no una nueva selección de modelo.
#
# Estimado: ~3 evaluaciones por semilla. Con 500 archivos/familia y 5 pliegues, alrededor
# de 40-50 min por semilla => unas 3-4 h las cinco. Los CSV se escriben al final; si hace
# falta acortar, bajar --semillas a 0,1,2.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2d.sh"
    exit 1
fi

python3.11 -u exp2d_nombre_extension.py "$DATOS" \
    --por-familia 500 \
    --semillas 0,1,2,3,4 \
    --folds 5

echo "Fin: $(date)"
