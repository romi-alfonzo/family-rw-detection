#!/bin/bash
#SBATCH --job-name=tesis_diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --nodelist=c2
#SBATCH --time=1:00:00
#SBATCH --output=slurm-diag-%j.out
# Diagnóstico SIN aprendizaje de las cuatro familias que la configuración canónica no resuelve
# (NOTPETYA, JIGSAW, CRYPTOLOCKER, DARKSIDE) más WASTEDLOCKER y SUNCRYPT. Mira prefijos y
# sufijos de 16 bytes (en total y por tipo de documento), colisiones entre familias, resto del
# tamaño módulo 16 y un perfil de entropía de hasta 32 bloques. Cuenta además cuántos de los
# 310 .pdf devueltos al corpus empiezan con firma PDF en claro.
#
# Es para diseñar el próximo rasgo DESDE LOS DATOS: en 2e, 2e-b y 2e-c el mecanismo razonado
# falló las tres veces. Un proceso, lectura liviana: ~5-10 min.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_diagnostico_dificiles.sh"
    exit 1
fi
python3.11 -u diagnostico_dificiles.py "$DATOS"
echo "Fin: $(date)"
