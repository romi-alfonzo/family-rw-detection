#!/bin/bash
#SBATCH --job-name=tesis_2f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=8:00:00
#SBATCH --output=slurm-exp2f-%j.out
# Exp. 2f -- EL SISTEMA COMPLETO del frente de archivos: bytes + estructura + forma del nombre,
# y una cuarta columna que suma la extensión literal. Cada capa se había medido de a pares
# contra los bytes (bytes+nombre 0,9998 en 2d; bytes+estructura 0,936 en 2e) pero nunca todas
# juntas. Criterio de Romina: las técnicas son una secuencia, no alternativas.
#
# Además: F1 por familia en las CINCO semillas (no solo una), y validación sobre tipos de
# documento nunca vistos, para que el número final tenga el respaldo de los anteriores.
#
# QUÉ MIRAR: la tabla (A) de F1 por familia —NOTPETYA, JIGSAW, CRYPTOLOCKER, DARKSIDE arriba—,
# la línea «Familias con F1 medio < 0,99», y el VEREDICTO. Si F4 falla, la forma del nombre
# codifica en parte el tipo de documento y hay que declararlo junto al número.
#
# Estimado: ~30 min por semilla (la extracción de rasgos es lo lento) x 5 + ~35 min de la
# validación por tipos => ~3 h. El --time de 8 h deja margen.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2f.sh"
    exit 1
fi
python3.11 -u exp2f_sistema_completo.py "$DATOS" --por-familia 500 --semillas 0,1,2,3,4 --folds 5
echo "Fin: $(date)"
