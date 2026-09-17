#!/bin/bash
#SBATCH --job-name=tesis_grafo_marc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --nodelist=c2
#SBATCH --output=slurm-grafo-%j.out
# B.3: grafo de marcadores compartidos entre plantillas + protocolo P3.
# Liviano: solo trabaja sobre 144 notas de texto. Lo caro son las 4 evaluaciones de P3
# (2 pliegues x 10 semillas cada una), que son unos 100 ajustes de LinearSVC => minutos.
# El corpus se espera en ./corpus_v2 (el script lo detecta solo cuando esta todo plano).

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Nucleos asignados: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

python3.11 -u grafo_marcadores.py

echo "Fin: $(date)"
