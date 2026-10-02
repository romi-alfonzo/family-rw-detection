#!/bin/bash
#SBATCH --job-name=tesis_exp2e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=c2
#SBATCH --time=6:00:00
#SBATCH --output=slurm-exp2e-%j.out
# Exp. 2e -- MEJORAR el frente de archivos con rasgos ESTRUCTURALES. Sin nombre ni extensión:
# la mejora, si aparece, es del contenido y no arrastra la limitación de campaña del Exp. 2d.
#
# POR QUÉ ESTOS RASGOS. El job 3630 midió que el 79,6 % de la importancia está en la cola, y
# que SUNCRYPT tiene entropía de cola 4,78 y NOTPETYA 6,58 contra 7,44 del resto: las dos SÍ
# dejan algo estructurado al final y aun así no se identifican. El capítulo ya explica por
# qué: ese bloque «varía en cada archivo». La representación posicional aprende VALORES de
# byte en posiciones fijas, así que un pie cuyo contenido cambia es invisible para ella
# aunque su PRESENCIA, su TAMAÑO y su ALEATORIEDAD sean constantes en la familia. Los 44
# rasgos describen eso: entropía a ocho profundidades por extremo, ocho bloques repartidos,
# largo del bloque final no aleatorio, tamaño y sus restos módulo 16/512/4096, y estadísticos
# de la distribución de bytes.
#
# TRES COLUMNAS sobre la MISMA partición:
#   (1) bytes canónico ....... 512+512 posicional, la referencia del Exp. 2c
#   (2) bytes + estructura ... la propuesta
#   (3) solo estructura ...... control: ¿alcanzan solos o son complementarios?
#
# CORPUS CORREGIDO: `es_documentacion()` devuelve los 143 .pdf cifrados de BADRABBIT y los
# 167 de NOTPETYA que el filtro por extensión descartaba. La columna (1) NO tiene por qué
# reproducir el 0,912 publicado, y cuánto se movió es parte del resultado. El script lo
# informa; no aborta.
#
# QUÉ MIRAR EN EL LOG, en este orden:
#   1. «Control de integridad» y el listado de archivos en claro. JIGSAW debería aparecer con
#      1 o 2 (los .pdf que la cuarentena del 22-09 no alcanzó); CERBER con ~500 (cifrado
#      parcial, ya declarado).
#   2. «Efecto del CORPUS CORREGIDO»: cuánto se movió la columna canónica contra 0,9120.
#   3. «DÓNDE CAE LA MEJORA»: el delta por familia. Si cae en las seis difíciles, el
#      experimento cierra el argumento del capítulo; si se reparte, es otra cosa.
#
# Estimado: ~3 min de rasgos + ~3 min de bosques por semilla => 30-45 min. El --time de 6 h
# sobra. Nodo: c1 y c3 tienen /scratch degradado.

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Nodo: $(hostname) | Núcleos: $SLURM_CPUS_PER_TASK | Inicio: $(date)"

if [ -z "$DATOS" ]; then
    echo "ERROR: falta la variable DATOS. Lanzar así:"
    echo "  DATOS=/scratch/ralfonzo/Napierone-small sbatch --nodelist=c2 --export=ALL,DATOS job_exp2e.sh"
    exit 1
fi

python3.11 -u exp2e_estructura_bytes.py "$DATOS" \
    --por-familia 500 \
    --semillas 0,1,2,3,4 \
    --folds 5

echo "Fin: $(date)"
