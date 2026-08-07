#!/bin/bash
#SBATCH --job-name=tesis_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-smoke-%j.out
# Prueba rápida (~2 min): verifica entorno, corpus y que el pipeline corre.
# CORRER ESTE PRIMERO, antes de encolar los trabajos largos.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "=== Entorno ==="
hostname; date
python3.11 --version
echo "Núcleos asignados: $SLURM_CPUS_PER_TASK"
echo "site-packages del usuario: $(python3.11 -u -m site --user-site)"
python3.11 -u -c "import sklearn, numpy; print('sklearn', sklearn.__version__, '| numpy', numpy.__version__)"
python3.11 -u -c "import pandas; print('pandas', pandas.__version__)"
python3.11 -u -c "
try:
    import bs4; print('bs4', bs4.__version__)
except ImportError:
    print('bs4 NO instalado -> se usa el fallback regex (OK, sin pérdida)')
"

echo
echo "=== Carga del corpus ==="
python3.11 -u -c "
from pathlib import Path
from clasificador_notas_v2 import cargar_corpus, CORPUS_DIR, OUT_DIR, N_JOBS
t, y, a, m = cargar_corpus(Path(CORPUS_DIR))
print('CORPUS_DIR:', CORPUS_DIR)
print('OUT_DIR   :', OUT_DIR)
print('N_JOBS    :', N_JOBS)
print(f'Notas: {len(t)} | Familias: {len(set(y))}')
vac = [x for x in t if len(x.strip()) < 10]
print('Notas vacias (debe ser 0):', len(vac))
"

echo
echo "=== Smoke del gridsearch (grilla mínima) ==="
python3.11 -u gridsearch_notas.py --smoke

echo "Fin: $(date)"
