#!/bin/bash
#SBATCH --job-name=tesis_rev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=5:00:00
#SBATCH --output=slurm-revision-%j.out
# Revisión independiente de la tesis -- frente de archivos, en UNA sola corrida:
#   (A) censo independiente del conjunto completo (recuentos, cabeceras en claro, extensiones,
#       duplicados SHA-256)
#   (B) validación cruzada con rasgos REIMPLEMENTADOS desde la descripción del capítulo 4
#       (bytes · +estructura · +extensión · bytes+extensión · solo extensión), 5 semillas
#   (C) dejar-un-tipo-fuera con ponderación de clases, 5 semillas para el sistema completo
#   (D) controles de fuga: duplicados entre pliegues y tabla de consulta por extensión
#
# QUÉ MIRAR: el bloque «VEREDICTO DEL PREREGISTRO» al final del log, y la línea «NIVEL».
# Estimado: 2 a 3 h (el censo lee el conjunto entero: unos minutos).

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_revision_tesis.sh"
    exit 1
fi
python3.11 -u revision_tesis_archivos.py "$DATOS" --por-familia 500 --semillas 0,1,2,3,4 --folds 5
echo "Fin: $(date)"
