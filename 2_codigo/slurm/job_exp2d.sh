#!/bin/bash
#SBATCH --job-name=tesis_exp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=12:00:00
#SBATCH --output=slurm-exp2d-%j.out
# Experimento 2d (aporte del NOMBRE y de la EXTENSIÓN sobre los bytes) + A.3 (curva de
# aprendizaje). Los dos elementos pendientes del frente de archivos que pidió el tutor en
# la reunión del 12-08-2026, en un solo job porque comparten la carga de datos.
#
# 2d -- tres columnas sobre la MISMA partición, para que sean comparables:
#   (1) solo bytes ................... la configuración canónica del Exp. 2c
#   (2) + forma del nombre ........... defendible: longitud, composición, entropía
#   (3) + extensión literal .......... COTA SUPERIOR declarada, no un método
#
# Antes de entrenar, el script MIDE si la extensión determina la etiqueta en este
# conjunto (cuenta extensiones por familia y evalúa una tabla de consulta). Si esa
# tabla ya acierta casi todo, la columna (3) se reporta como memorización de un
# diccionario y no como capacidad de identificar familias.
#
# A.3 -- exactitud y macro-F1 contra archivos por familia (10 a 500), subconjuntos
# anidados, solo bytes, con el delta pareado entre tamaños consecutivos e IC 95 %.
# Contesta «¿cuántas muestras hacen falta?» en el frente de archivos, como B.1 en notas.
#
# No repite la búsqueda de hiperparámetros: usa los del Exp. 2c (HIPER_2C en
# clasificador_bytes.py: 300 árboles, profundidad 20, hoja 2, 0,3 de características),
# porque lo que se mide es el aporte de las características y no una nueva selección de
# modelo. Si estos valores no coinciden con los de clasificador_bytes.py, la columna (1)
# deja de reproducir la referencia publicada y el experimento pierde sentido.
#
# ANTES DE LANZAR: verificar que CERBER-small/_sin_cifrar/ existe en el clúster. Este
# experimento usa el NOMBRE, así que los 12 JPEG en claro con su nombre original harían
# trivial la columna (2). El script los detecta por magia de tipo y lo avisa en el log.
#
# Estimado: 15 evaluaciones de 2d (3 columnas x 5 semillas) a ~10 min cada una, más la
# curva (~1,5 h) => 4-5 h. El --time de 12 h deja margen. Si hay que acortar:
# --semillas 0,1,2 y --tamanos 25,100,250,500.
# Nodo: c1 y c3 tienen /scratch degradado (ver PENDIENTE_REDACCION.md, sección G).

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2d.sh"
    exit 1
fi

python3.11 -u exp2d_nombre_extension.py "$DATOS" \
    --por-familia 500 \
    --semillas 0,1,2,3,4 \
    --folds 5 \
    --tamanos 10,25,50,100,200,350,500 \
    --semillas-curva 0,1,2

echo "Fin: $(date)"
