#!/bin/bash
#SBATCH --job-name=tesis_bytes_ms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-bytesms-%j.out
# A.2 del plan de mejoras: DISPERSIÓN del frente de archivos.
#
# El tutor pidió el 12-08-2026 reportar el desvío también en archivos, que hasta ahora
# iba sin error mientras las notas se reportan como media ± desvío sobre 10 semillas.
# Se usan LAS MISMAS diez semillas que clasificador_notas_v2.py (0-9), de modo que los
# dos frentes queden comparables en la tesis.
#
# Repite SOLO la evaluación final (posicional + RandomForest, 500 archivos/familia,
# 5 pliegues) con los hiperparámetros ya elegidos por la búsqueda anidada del Exp. 2c.
# Declararlo así: mide la dispersión de la estimación, no una nueva selección de modelo.
#
# Estimado: ~12-15 min por semilla => 2-2,5 h las diez. Los CSV se reescriben en cada
# semilla, así que un corte por tiempo no pierde lo ya corrido.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_bytes_multisemilla.sh"
    exit 1
fi

python3.11 -u clasificador_bytes.py "$DATOS" \
    --multisemilla 0,1,2,3,4,5,6,7,8,9 \
    --por-familia-final 500 \
    --folds-finales 5

echo "Fin: $(date)"
