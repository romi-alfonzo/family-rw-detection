#!/bin/bash
#SBATCH --job-name=tesis_2eb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=3:00:00
#SBATCH --output=slurm-exp2eb-%j.out
# Exp. 2e-b -- ¿QUÉ rasgo estructural hace el trabajo? Diagnóstico del job 4058.
#
# El 2e mejoró de 0,9114 a 0,9359 de macro-F1 y la mejora cayó entera en las seis difíciles
# (+0,1186 contra +0,0006). Pero la predicción sobre el MECANISMO falló: subieron más
# WASTEDLOCKER y DARKSIDE, que están «exactamente en el techo» de entropía, que SUNCRYPT y
# NOTPETYA, para las que se diseñó el rasgo del pie poco aleatorio. Si no hay estructura de
# entropía que medir, la señal viene de otro lado, y la hipótesis que queda es el TAMAÑO.
#
# Mide dos cosas: (A) importancias del bosque sobre bytes+estructura, y (B) ablación por
# grupo sobre solo-estructura, con validación cruzada, reportando aparte la caída en las
# seis difíciles. La ablación manda sobre las importancias porque los rasgos están
# correlacionados: ocho entropías de cola miden casi lo mismo y se reparten el crédito.
#
# QUÉ MIRAR: la columna «caída dif.» de la tabla (B), y la línea «SOLO los 5 rasgos de
# tamaño». Si con cinco números de tamaño las seis difíciles ya suben, la explicación del
# capítulo es el relleno a bloque y el pie de longitud fija, no la entropía.
#
# Una semilla alcanza: es diagnóstico, no una cifra reportable. Estimado 25-40 min.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2e_rasgo.sh"
    exit 1
fi
python3.11 -u exp2e_que_rasgo.py "$DATOS" --por-familia 500 --semilla 0 --folds 5
echo "Fin: $(date)"
