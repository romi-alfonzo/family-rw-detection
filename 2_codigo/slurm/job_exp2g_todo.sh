#!/bin/bash
#SBATCH --job-name=tesis_2g_todo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodelist=c2
#SBATCH --time=20:00:00
#SBATCH --output=slurm-exp2g-todo-%j.out
# Exp. 2g sobre TODOS los archivos de NapierOne-small (verificación pedida por Romina el 2026-10-29):
# el sistema completo (bytes de cabecera y cola + rasgos estructurales + forma de la extensión), con las
# mismas semillas, pliegues e hiperparámetros del job 4091, pero sin muestrear 500 archivos por familia.
#
# --por-familia 100000 toma todos los archivos de cada familia (el script recorta la permutación al
# tamaño pedido, y si pide más de los que hay, se queda con todos). Con todos los archivos, las cinco
# semillas solo cambian el reparto de los pliegues.
#
# QUÉ MIRAR: cv_resumen.csv (exactitud y macro-F1 de las cinco semillas) y cv_por_familia_resumen.csv,
# contra el 0,9998 ± 0,0001 del job 4091 con 500 por familia.
#
# Estimado: el job 4091 (15.000 archivos) tardó unas 3 h; con el doble de archivos, entre 6 y 10 h.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2g_todo.sh"
    exit 1
fi
python3.11 -u exp2g_nombre_robusto.py "$DATOS" --por-familia 100000 --semillas 0,1,2,3,4 --folds 5
echo "Fin: $(date)"
