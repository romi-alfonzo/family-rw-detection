#!/bin/bash
#SBATCH --job-name=tesis_curva_notas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodelist=c2
#SBATCH --output=slurm-curva-%j.out
# B.1: curva de aprendizaje del frente de notas.
# Esta SI es cara: unos 3.600 ajustes de TF-IDF char_wb 3-5 + LinearSVC (1-2,5 s cada
# uno). En una PC de 8 nucleos son ~20 min. El script paraleliza POR REPETICION y respeta
# SLURM_CPUS_PER_TASK, asi que aca usa los 8 nucleos del job y no los del nodo entero.
#
# Este es el job que hay que volver a lanzar DESPUES DE CADA LOTE DE NOTAS NUEVAS: la
# curva es el medidor de si la recoleccion esta rindiendo.
#
# Antes de la corrida completa hace el control de correccion (el punto k="todo" tiene que
# reproducir el evaluador canonico); si no coincide, aborta solo.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Nucleos asignados: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

python3.11 -u curva_aprendizaje_notas.py

echo "Fin: $(date)"
