#!/bin/bash
#SBATCH --job-name=tesis_tipos_pond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=4:00:00
#SBATCH --output=slurm-tipos-pond-%j.out
# Validación por tipos de documento con class_weight="balanced", el modelo de la validación
# cruzada y de la validación por tipos publicada del Exp. 2c. Las del 2e-c, 2f y 2g se corrieron
# sin ponderación (error detectado el 2026-09-28), y en el pliegue jpg eso deja a BLACKMATTER
# --que en NapierOne-small es solo imágenes-- con F1 0.
#
# QUÉ MIRAR: la PUERTA (tiene que decir ✔), la tabla por pliegue, «BLACKMATTER en el pliegue
# jpg» y el VEREDICTO.
#
# Estimado: carga ~10 min + 4 columnas x 7 pliegues => ~1 h.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_tipos_ponderada.sh"
    exit 1
fi
python3.11 -u validacion_tipos_ponderada.py "$DATOS" --por-familia 500
echo "Fin: $(date)"
