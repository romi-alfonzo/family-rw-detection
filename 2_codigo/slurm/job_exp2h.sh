#!/bin/bash
#SBATCH --job-name=tesis_2h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=6:00:00
#SBATCH --output=slurm-exp2h-%j.out
# Exp. 2h -- cierre del frente de archivos: todas las dudas abiertas en una sola corrida.
#   (A) censo del conjunto vigente y duplicados por SHA-256
#   (B) dejar-un-tipo-fuera con ponderación de clases (la de 2e-c/2f/2g se corrió sin ponderar)
#   (C) ablación del sistema completo en validación cruzada (¿qué aporta cada capa?)
#   (D) el sistema completo bajo tipo no visto con cinco semillas de muestreo
#
# El log sale en orden de importancia: (A) en minutos, (B) en ~40 min, (C) y (D) después.
# QUÉ MIRAR: las PUERTAS (✔), la tabla de (B), «BLACKMATTER en el pliegue jpg», el resumen de
# la ablación y el VEREDICTO al final.
#
# Estimado: ~2,5 h.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2h.sh"
    exit 1
fi
python3.11 -u exp2h_cierre_archivos.py "$DATOS" --por-familia 500 --semillas 0,1,2,3,4
echo "Fin: $(date)"
