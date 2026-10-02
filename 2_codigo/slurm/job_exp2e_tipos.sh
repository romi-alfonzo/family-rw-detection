#!/bin/bash
#SBATCH --job-name=tesis_2ec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=4:00:00
#SBATCH --output=slurm-exp2ec-%j.out
# Exp. 2e-c -- dejar-un-tipo-fuera sobre la configuración canónica NUEVA (bytes + estructura).
#
# Romina decidió el 28-09 que el 0,936 del Exp. 2e sea el canónico del frente de archivos.
# El 0,912 al que reemplaza pasó por dejar-un-tipo-fuera (§4.5.5: entrenar sin un tipo de
# documento y evaluar sobre él, promedio 0,879, caída 0,031), que es lo que descarta que el
# clasificador aprenda el documento de origen. El 0,936 no pasó por eso, y tiene un riesgo
# concreto: los rasgos de tamaño correlacionan con el tipo. Este job lo mide.
#
# Réplica exacta de los pliegues del 2c, con DOS representaciones sobre cada uno para que el
# delta sea pareado: solo bytes (referencia re-medida sobre la base actual) y bytes+estructura.
#
# QUÉ MIRAR: la tabla por pliegue (columna Δ F1), y el VEREDICTO al final. Si P2 y P4 cumplen,
# el 0,936 pasa la misma prueba que el 0,912. Si el Δ promedio es negativo, el canónico vuelve
# a ser el 0,912 y hay que decirlo.
#
# Estimado: ~20 min de carga y rasgos + 7 pliegues x 2 ajustes de ~2 min => ~50 min.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"
if [ -z "$DATOS" ]; then
    echo "ERROR: falta DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2e_tipos.sh"
    exit 1
fi
python3.11 -u exp2e_validacion_tipos.py "$DATOS" --por-familia 500 --semilla 0
echo "Fin: $(date)"
