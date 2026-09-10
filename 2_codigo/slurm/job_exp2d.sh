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
# 2d -- cinco columnas sobre la MISMA partición, para que sean comparables:
#   (1) solo bytes ................... la configuración canónica del Exp. 2c
#   (2) + forma del nombre ........... longitud, composición, entropía
#   (3) + extensión literal .......... COTA SUPERIOR declarada, no un método
#   (0a) solo forma del nombre ....... CONTROL, sin un solo byte del contenido
#   (0b) solo extensión literal ...... CONTROL, sin un solo byte del contenido
#
# Antes de entrenar, el script MIDE si la extensión determina la etiqueta en este
# conjunto (cuenta extensiones por familia y evalúa una tabla de consulta). Si esa
# tabla ya acierta casi todo, la columna (3) se reporta como memorización de un
# diccionario y no como capacidad de identificar familias.
#
# Las dos columnas de control se agregaron después del job 3771, que dio (2) = 0,9985 de
# macro-F1 contra 0,8983 de (1): un salto de +0,10 que hay que poder interpretar. El
# diagnóstico previo mide la circularidad de la extensión LITERAL, pero no la de la FORMA
# del nombre, y la forma es huella de campaña igual que la extensión. Sin (0a) no se puede
# distinguir «los bytes ayudados por el nombre» de «el nombre con los bytes al lado».
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
# Estimado: 25 evaluaciones de 2d (5 columnas x 5 semillas), más la curva. En el job 3771
# las tres columnas de bytes tardaron 234 s por semilla; las dos de control son más
# baratas. => 1-2 h de 2d más ~1,5 h de curva. El --time de 12 h deja margen de sobra.
# Si hay que acortar: --semillas 0,1,2 y --tamanos 25,100,250,500.
#
# El job 3771 se canceló en la semilla 4 y perdió las cuatro semillas ya medidas, porque
# los CSV se escribían recién al terminar el bucle. Ahora el script reescribe
# `exp2d_por_semilla.csv` al final de cada semilla, así que una interrupción deja en disco
# todo lo que alcanzó a medir.
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
