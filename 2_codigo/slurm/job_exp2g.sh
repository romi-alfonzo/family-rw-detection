#!/bin/bash
#SBATCH --job-name=tesis_2g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=8:00:00
#SBATCH --output=slurm-exp2g-%j.out
# Exp. 2g -- el sistema completo con rasgos de nombre ROBUSTOS.
#
# El 2f dio 0,9998 pero su validación por tipos colapsó en jpg (0,8052 -> 0,2167 al sumar el
# nombre). Causa verificada mirando los archivos: la base del nombre la puso NapierOne, y en los
# jpg lleva un «-fromweb» (0001-jpg-fromweb.jpg.avos2 contra 0001-pdf.pdf.avos2). La forma del
# nombre completo aprendió esa herencia. El arreglo: rasgos solo sobre la extensión final, lo
# que agrega el ransomware.
#
# QUÉ MIRAR: la tabla (B), columna Δ(5)-(2) en la fila jpg -- si ya no colapsa, está resuelto --,
# el F1 por familia de las cuatro difíciles, y el VEREDICTO.
#
# Estimado: ~25 min por semilla x 5 + ~50 min de tipos no vistos => ~3 h.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2g.sh"
    exit 1
fi
python3.11 -u exp2g_nombre_robusto.py "$DATOS" --por-familia 500 --semillas 0,1,2,3,4 --folds 5
echo "Fin: $(date)"
