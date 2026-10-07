#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cifras_finales.py -- TODAS las cifras del frente de notas, con su fuente, verificadas.

PARA QUE SIRVE. Cada numero que este trabajo reporta sale de una corrida concreta, guardada en
`4_resultados/`. Este script recorre esas salidas, **busca cada cifra en su archivo de origen** y
avisa si alguna no aparece donde deberia. No recalcula nada: verifica que lo escrito coincida con
lo medido.

COMO SE USA. `python cifras_finales.py` imprime el listado completo agrupado por base, con la
fuente de cada numero y el resultado de la verificacion. `--solo-fallas` muestra unicamente lo
que no verifica.

LAS DOS BASES NO SE MEZCLAN. Sus cifras no son comparables y cada una se declara por separado:
  BASE A -- nucleo canonico: 149 notas, 99 plantillas, 30 familias, P2bal, 50 semillas.
            Es la base de la tesis.
  BASE B -- corpus extendido: 596 notas, 106 familias, 20 semillas. Experimento aparte, con
            etiquetas de las fuentes y sin auditoria de procedencia.

METRICA PEGADA. En este proyecto conviven exactitud, exactitud balanceada, macro-F1 sobre 30 y
sobre 28 familias, MCC, cobertura y acierto-donde-aplica. Ningun numero se reporta sin decir cual
es y sobre que base se midio.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "4_resultados"

# (grupo, descripcion, metrica, valor, archivo fuente)
CIFRAS = [
    # ---------------- BASE A: cifra de cabecera ----------------
    ("A · cabecera", "cascada", "macro-F1 (30 familias)", "0.7417", "_log_p2bal_149.txt"),
    ("A · cabecera", "cascada", "IC 95 % entre semillas", "[0.7328; 0.7505]", "_log_p2bal_149.txt"),
    ("A · cabecera", "cascada", "macro-F1 (28 evaluables)", "0.7946", "_log_p2bal_149.txt"),
    ("A · cabecera", "cascada", "exactitud", "0.8123", "_log_p2bal_149.txt"),
    ("A · cabecera", "cascada", "exactitud balanceada", "0.7798", "_log_p2bal_149.txt"),
    ("A · cabecera", "cascada", "MCC", "0.8042", "_log_p2bal_149.txt"),
    ("A · cabecera", "texto solo", "macro-F1 (30 familias)", "0.6551", "_log_p2bal_149.txt"),
    ("A · cabecera", "texto solo", "macro-F1 (28 evaluables)", "0.7019", "_log_p2bal_149.txt"),
    ("A · cabecera", "texto solo", "exactitud", "0.7191", "_log_p2bal_149.txt"),
    ("A · cabecera", "texto solo", "MCC", "0.7092", "_log_p2bal_149.txt"),
    ("A · cabecera", "P2bal vs P2, cascada", "delta pareado", "+0.2225", "_log_p2bal_149.txt"),
    ("A · cabecera", "familias sin train por pliegue, P2", "control", "3.86", "_log_p2bal_149.txt"),
    ("A · cabecera", "familias sin train por pliegue, P2bal", "control", "1.00", "_log_p2bal_149.txt"),

    # ---------------- BASE A: los tres protocolos, 50 semillas ----------------
    # Es la corrida de la que salen el 83,9 % y el 0,789 de la tabla comparativa.
    ("A · protocolos", "P1 plantilla ya catalogada", "exactitud", "0.8389", "_log_lemmou_149.txt"),
    ("A · protocolos", "P1 plantilla ya catalogada", "macro-F1", "0.7889", "_log_lemmou_149.txt"),
    ("A · protocolos", "P1 plantilla ya catalogada", "exactitud balanceada", "0.8002", "_log_lemmou_149.txt"),
    ("A · protocolos", "P2 plantilla nunca vista", "exactitud", "0.5785", "_log_lemmou_149.txt"),
    ("A · protocolos", "P2 plantilla nunca vista", "macro-F1", "0.4593", "_log_lemmou_149.txt"),
    ("A · protocolos", "L esquema de la literatura previa", "exactitud", "0.8389", "_log_lemmou_149.txt"),
    ("A · protocolos", "L esquema de la literatura previa", "macro-F1", "0.7767", "_log_lemmou_149.txt"),
    ("A · protocolos", "L, vecina de la MISMA plantilla", "acierto", "0.9589", "_log_lemmou_149.txt"),
    ("A · protocolos", "L, vecina de OTRA plantilla", "acierto", "0.7237", "_log_lemmou_149.txt"),

    # ---------------- BASE A: el mismo P1 y P2 estimados con 10 semillas ----------------
    # Convive con el bloque anterior a proposito: son la misma configuracion con otro numero
    # de semillas, y en la tesis aparecen las dos. Por eso cada tabla declara la suya.
    ("A · 10 semillas", "P1 combinado + LinearSVC", "macro-F1", "0.799", "_log_clasificador_149.txt"),
    ("A · 10 semillas", "P1 combinado + LinearSVC", "exactitud", "0.842", "_log_clasificador_149.txt"),
    ("A · 10 semillas", "P2 combinado + LinearSVC", "macro-F1", "0.468", "_log_clasificador_149.txt"),
    ("A · 10 semillas", "P2 combinado + LinearSVC", "exactitud", "0.580", "_log_clasificador_149.txt"),

    # ---------------- BASE 146: el corpus inicial, que el capitulo conserva declarado ----------------
    # NO se corrigen en la tesis: tienen otra base, no un error. El valor guardado es el de la
    # corrida; entre parentesis, como lo cita el documento ya redondeado.
    ("146 · corpus inicial", "P1 caracteres + LinearSVC (cita 0,818)", "exactitud", "0.8178", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P1 caracteres + LinearSVC (cita 0,760)", "macro-F1", "0.7596", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P1 caracteres + LinearSVC (cita 0,777)", "exactitud balanceada", "0.7765", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P1 caracteres + LinearSVC (cita 0,798)", "weighted-F1", "0.7981", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P2 combinado + LinearSVC (cita 0,551)", "exactitud", "0.5513", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P2 combinado + LinearSVC (cita 0,435)", "macro-F1", "0.4353", "resultados_canonicos/corrida_canonica_resumen.csv"),
    ("146 · corpus inicial", "P2 combinado + LinearSVC (cita 0,512)", "weighted-F1", "0.5120", "resultados_canonicos/corrida_canonica_resumen.csv"),

    # ---------------- BASE A: la cascada con PLANTILLA YA CATALOGADA (P1, 50 semillas) ----------------
    # Medicion del 2026-10-01, preregistro commiteado en 7be8db6 ANTES de correr. Hasta ese dia
    # todas las cifras de P1 del documento eran del clasificador de texto solo.
    ("A · P1 cascada", "cascada, plantilla ya catalogada", "exactitud", "0.8866", "_log_m3_149_P1.txt"),
    ("A · P1 cascada", "cascada, plantilla ya catalogada", "macro-F1", "0.8592", "_log_m3_149_P1.txt"),
    ("A · P1 cascada", "capa de reglas bajo P1", "notas que resuelve (de 149)", "95.3", "_log_m3_149_P1.txt"),
    ("A · P1 cascada", "capa de reglas bajo P1", "acierto", "0.9953", "_log_m3_149_P1.txt"),
    ("A · P1 cascada", "texto solo bajo P1 (puerta de entrada)", "exactitud", "0.8389", "_log_m3_149_P1.txt"),
    ("A · P1 cascada", "texto solo bajo P1 (puerta de entrada)", "macro-F1", "0.7889", "_log_m3_149_P1.txt"),

    # Los macro-F1 de P1cat (17 familias) y de P2bal-28 sobre 145 notas se SACARON el 2026-10-01:
    # f1_score sin labels= promedia sobre la union de etiquetas verdaderas y predichas, y cada
    # familia de afuera que recibe una prediccion entra con F1 0 (reparo 3 de la revision).
    # No se citan. Las exactitudes de esos subconjuntos no tienen el problema y quedan.
    # ---------------- P1cat: plantilla GARANTIZADA en el catalogo (50 semillas) ----------------
    # Replica controlada del 2026-10-01, preregistro commiteado en 5502d75 antes de correr.
    # Evalua 73 notas de 17 familias (las que tienen alguna plantilla repetida). La exactitud
    # es comparable; el macro-F1 NO, porque promedia solo esas 17 familias.
    ("P1cat", "cascada, plantilla en el catalogo", "exactitud", "0.9740", "_log_p1_catalogada.txt"),
    ("P1cat", "texto solo, plantilla en el catalogo", "exactitud", "0.9479", "_log_p1_catalogada.txt"),
    ("P1cat", "capa de reglas", "cobertura", "0.8551", "_log_p1_catalogada.txt"),
    ("P1cat", "capa de reglas", "acierto", "1.0000", "_log_p1_catalogada.txt"),

    # ---------------- Plantilla NUNCA vista sobre las 28 familias evaluables (50 semillas) ----------------
    # p2bal_evaluables.py, preregistro en 4f23e4c. 145 notas (sin BADRABBIT ni CRYPTOLOCKER).
    # El macro-F1 de esta fila se calcula sobre las 145 notas; el PUBLICADO sobre 28 (0,7946)
    # usa labels = evaluables sobre las 149. Son dos convenciones: al citar, decir cual.
    ("P2bal-28", "cascada, plantilla nunca vista", "exactitud (145 notas)", "0.8348", "_log_p2bal_evaluables.txt"),
    ("P2bal-28", "texto solo, plantilla nunca vista", "exactitud (145 notas)", "0.7389", "_log_p2bal_evaluables.txt"),
    ("P2bal-28", "error de la cascada", "fraccion del texto", "0.9758", "_log_p2bal_evaluables.txt"),
    ("P2bal-28", "error de la cascada", "fraccion en parejas de linaje", "0.4207", "_log_p2bal_evaluables.txt"),

    # ---------------- PAREADO: las mismas 69 notas con y sin su plantilla en el catalogo ----------------
    # pareado_conocida_vs_nueva.py, preregistro en 453e579. 69 notas de 15 familias, 50 semillas.
    ("Pareado", "cascada, plantilla conocida", "exactitud", "0.9725", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "cascada, plantilla nueva", "exactitud", "0.8484", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "cascada", "diferencia de exactitud", "+0.1241", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "cascada", "IC 95 % de la diferencia", "[+0.1067; +0.1414]", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "texto solo, plantilla conocida", "exactitud", "0.9449", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "texto solo, plantilla nueva", "exactitud", "0.7299", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "texto solo", "diferencia de exactitud", "+0.2151", "_log_pareado_conocida_vs_nueva.txt"),
    ("Pareado", "texto solo", "IC 95 % de la diferencia", "[+0.1917; +0.2385]", "_log_pareado_conocida_vs_nueva.txt"),

    # Control de tamano igualado (pareado_tamano_igualado.py, preregistro d24ff7b): entrenamiento
    # de «conocida» recortado de 112,5 a 74,5 notas, igual que «nueva».
    ("Pareado igualado", "cascada, plantilla conocida, tamano igualado", "exactitud", "0.9617", "_log_pareado_tamano_igualado.txt"),
    ("Pareado igualado", "cascada", "diferencia de exactitud", "+0.1133", "_log_pareado_tamano_igualado.txt"),
    ("Pareado igualado", "texto solo", "diferencia de exactitud", "+0.2104", "_log_pareado_tamano_igualado.txt"),

    # Contraste limpio (contraste_limpio.py, preregistro 52f695c): las mismas 69 notas, dejando
    # afuera solo la nota o toda su plantilla. Determinista, semilla 0. IC por plantillas.
    ("Contraste limpio", "cascada, conocida", "exactitud", "0.9855", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "cascada, nueva", "exactitud", "0.9130", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "cascada", "diferencia", "+0.0725", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "cascada", "IC 95 %", "[+0.0000; +0.2000]", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "texto solo", "diferencia", "+0.2029", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "texto solo", "IC 95 %", "[-0.0161; +0.4336]", "_log_contraste_limpio.txt"),
    ("Contraste limpio", "sin mixtas, cascada y texto", "diferencia", "+0.0909", "_log_contraste_limpio.txt"),

    # Mejoras de la cascada bajo P1 (mejoras_cascada_p1.py, preregistro e27b28a), 149 notas, 30 familias.
    ("Mejoras P1", "V2 normalizacion de URL", "exactitud", "0.8895", "_log_mejoras_cascada_p1.txt"),
    ("Mejoras P1", "V2 normalizacion de URL", "macro-F1", "0.8626", "_log_mejoras_cascada_p1.txt"),
    ("Mejoras P1", "V2 contra V0", "delta exactitud", "+0.0030", "_log_mejoras_cascada_p1.txt"),
    ("Mejoras P1", "V2 contra V0", "IC 95 % delta exactitud", "[+0.0014; +0.0045]", "_log_mejoras_cascada_p1.txt"),
    ("Mejoras P1", "V2 contra V0", "delta macro-F1", "+0.0034", "_log_mejoras_cascada_p1.txt"),
    ("Mejoras P1", "V1 capa de catalogo", "notas que decide por semilla", "8.6", "_log_mejoras_cascada_p1.txt"),

    # ---------------- EXTENSION: catalogo ampliado con copias de tria.ge (P1, 50 semillas) ----------------
    # p1_copias_triage.py, preregistro 33a68e3 (antes de recolectar). Prueba = las 149 notas; las 42
    # copias (6 locales + 36 de tria.ge) solo entran al entrenamiento. Base propia: NO reemplaza al 0,8866.
    ("Ext. tria.ge", "cascada con catalogo ampliado", "exactitud", "0.9148", "_log_p1_copias_triage.txt"),
    ("Ext. tria.ge", "cascada con catalogo ampliado", "macro-F1", "0.8921", "_log_p1_copias_triage.txt"),
    ("Ext. tria.ge", "C1 - V0", "delta exactitud", "+0.0282", "_log_p1_copias_triage.txt"),
    ("Ext. tria.ge", "C1 - V0", "IC 95 % delta exactitud", "[+0.0250; +0.0313]", "_log_p1_copias_triage.txt"),
    ("Ext. tria.ge", "C1 - V0", "delta macro-F1", "+0.0329", "_log_p1_copias_triage.txt"),
    ("Ext. tria.ge", "copias admitidas de tria.ge", "cantidad", "36", "_log_p1_copias_triage.txt"),

    # ---------------- CURVA BAJO P2bal: texto solo, tope por plantillas, 149 notas, 50 semillas ----------------
    ("Curva P2bal", "k=1", "macro-F1 texto solo", "0.5759", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=1", "plantillas en entrenamiento por familia", "plantillas train/fam 0.97", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=2", "macro-F1 texto solo", "0.6599", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=2", "plantillas en entrenamiento por familia", "plantillas train/fam 1.57", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=3", "macro-F1 texto solo", "0.6608", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=todo", "macro-F1 texto solo", "0.6551", "_log_curva_p2bal_149.txt"),
    ("Curva P2bal", "k=todo", "plantillas en entrenamiento por familia", "plantillas train/fam 1.68", "_log_curva_p2bal_149.txt"),
    # ---------------- DIAGNOSTICO DE P1 (cascada canonica, 149 notas x 50 semillas = 7450 decisiones) ----------------
    ("Diagnostico P1", "decisiones con hermana en entrenamiento", "exactitud de la cascada", "3015/3085 = 0.9773", "_log_diagnostico_P1_agregado.txt"),
    ("Diagnostico P1", "decisiones con hermana en entrenamiento", "proporcion", "41.4 % de las decisiones", "_log_diagnostico_P1_agregado.txt"),
    ("Diagnostico P1", "decisiones sin hermana", "exactitud de la cascada", "3590/4365 = 0.8225", "_log_diagnostico_P1_agregado.txt"),
    ("Mejoras P1", "V1 capa de catalogo contra V0", "delta exactitud", "+0.0000 [+0.0000; +0.0000]", "_log_mejoras_cascada_p1.txt"),

    # ---------------- EXT. TRIA.GE SIN APUNTAR, EXTENSION de P1 (Romina, 06-10) (p1_copias_sin_apuntar.py, preregistro 659b79d) ----------------
    # Prueba = las 149 notas; las 202 copias (6 locales + 196 de tria.ge) solo entran al entrenamiento.
    # No reemplaza al 0,8866 canonico de P1: va como extension, en resultados_notas_ampliacion.tex.
    ("Ext. tria.ge sin apuntar (extension)", "cascada con catalogo ampliado (C1)", "exactitud", "0.9140 ± 0.0163", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "cascada con catalogo ampliado (C1)", "macro-F1", "0.8899 ± 0.0232", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "C1 - V0", "delta exactitud", "+0.0274", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "C1 - V0", "IC 95 % delta exactitud", "[+0.0233; +0.0315]", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "C1 - V0", "delta macro-F1", "+0.0307", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "C1 - V0", "IC 95 % delta macro-F1", "[+0.0257; +0.0358]", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "semillas en que C1 supera a V0", "cantidad", "C1 mejor en 48", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "decisiones con hermana en entrenamiento", "proporcion en C1", "57.5%", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "notas cuya plantilla recibe copia", "exactitud V0 -> C1", "40 notas: V0 0.8955 -> C1 1.0000", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "notas cuya plantilla no recibe copia", "exactitud V0 -> C1", "109 notas: V0 0.8833 -> C1 0.8824", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "copias en el catalogo (6 locales + 196 de tria.ge)", "cantidad", "Copias controladas: 202", "_log_p1_copias_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "recoleccion", "informes leidos", "1229 informes", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "recoleccion", "notas distintas", "681 notas distintas", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "recoleccion", "lecturas fallidas", "2 lecturas fallidas", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "inventario", "texto nuevo (no es copia)", "TEXTO_NUEVO: 275", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "inventario", "copia ambigua", "COPIA_AMBIGUA: 129", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "inventario", "repetida", "REPETIDA: 56", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "inventario", "copia con los mismos marcadores", "no admitidas: 24", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "inventario", "copias admitidas", "COPIA admitidas: 196, en 17 familias", "_log_inventario_sin_apuntar.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "errores por semilla sin ampliar (149 notas)", "cantidad", "V0 16.90", "_log_diagnostico_ampliacion_p1.txt"),
    ("Ext. tria.ge sin apuntar (extension)", "errores sin ampliar en las 76 notas unicas de su plantilla", "cantidad", "errores V0 14.28", "_log_diagnostico_ampliacion_p1.txt"),

    # ---------------- EXT. TRIA.GE, PLANTILLAS NUEVAS: la curva bajo P2bal con mas plantillas por familia ----------------
    # (p1_p2bal_plantillas_nuevas.py, preregistro ed85118; Romina lo llevo a la tesis el 06-10). Prueba = las 149 notas;
    # las 60 plantillas nuevas (N-rep) solo entrenan. 3,7 plantillas/familia = 1,68 + 60/30 (aritmetica, ver EXTERNAS).
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'plantillas nuevas, una nota por plantilla (N-rep)', 'cantidad', 'N-rep: 60 plantillas nuevas', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'familias que reciben alguna', 'cantidad', '23 familias reciben alguna', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'CRYPTOLOCKER: plantillas del corpus y nuevas (queda en 1)', 'cantidad', 'CRYPTOLOCKER                   1             0', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'NOTPETYA: plantillas del corpus y nuevas (queda en 2)', 'cantidad', 'NOTPETYA                       2             0', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, texto solo con N-rep', 'macro-F1', 'macro-F1 0.6498', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, texto solo, N-rep - V0', 'delta macro-F1 e IC 95 %', 'macro-F1 -0.0053 [-0.0136; +0.0031]', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, cascada con N-rep', 'exactitud', 'exactitud 0.7711', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, cascada con N-rep', 'macro-F1', 'macro-F1 0.7423', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, cascada, N-rep - V0', 'delta exactitud e IC 95 %', 'exactitud -0.0412 [-0.0489; -0.0335]', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, cascada, N-rep - V0', 'semillas peores', 'peor en 46', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'P2bal, cascada, N-rep - V0', 'delta macro-F1 e IC 95 %', 'macro-F1 +0.0007 [-0.0059; +0.0072]', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'F1 por familia MAZE, V0 -> N-rep', 'F1 de la familia', '0.592   0.953', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'F1 por familia MEDUZALOCKER, V0 -> N-rep', 'F1 de la familia', '0.731   0.834', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'F1 por familia PHOBOS, V0 -> N-rep', 'F1 de la familia', '0.768   0.392', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'F1 por familia DHARMA, V0 -> N-rep', 'F1 de la familia', '0.872   0.583', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'F1 por familia LOCKBIT, V0 -> N-rep', 'F1 de la familia', '0.699   0.476', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'plantillas nuevas cuya vecina es de otra familia', 'cantidad', '4 plantillas nuevas cuya nota', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'rotulo dudoso hacia LOCKBIT (RYUK)', 'coseno', 'RYUK~LOCKBIT 0.90', '_log_plantillas_nuevas.txt'),
    ('Ext. tria.ge plantillas nuevas (curva P2bal)', 'rotulo dudoso hacia LOCKBIT (RANSOMEXX)', 'coseno', 'RANSOMEXX~LOCKBIT 0.89', '_log_plantillas_nuevas.txt'),

    # ---------------- BASE A: incertidumbre ----------------
    ("A · incertidumbre", "cascada, remuestreo de plantillas", "IC 95 %", "0.6585", "_log_revision_bootstrap.txt"),
    ("A · incertidumbre", "cascada, remuestreo de plantillas", "IC 95 % (alto)", "0.8187", "_log_revision_bootstrap.txt"),
    ("A · incertidumbre", "texto, remuestreo de plantillas", "IC 95 %", "0.5654", "_log_revision_bootstrap.txt"),
    ("A · incertidumbre", "texto, remuestreo de plantillas", "IC 95 % (alto)", "0.7446", "_log_revision_bootstrap.txt"),

    # ---------------- BASE A: despliegue ----------------
    ("A · despliegue", "abstencion umbral 0,50", "cobertura", "0.7718", "_log_m3_149_p2bal.txt"),
    ("A · despliegue", "abstencion umbral 0,50", "acierto donde responde", "0.9324", "_log_m3_149_p2bal.txt"),
    ("A · despliegue", "capa de reglas", "acierto", "0.9928", "_log_m3_149_p2bal.txt"),
    ("A · despliegue", "capa de reglas", "notas que resuelve (de 149)", "80.3", "_log_m3_149_p2bal.txt"),
    ("A · despliegue", "capa de texto", "acierto sin umbral", "0.6025", "_log_m3_149_p2bal.txt"),
    ("A · despliegue", "lista de candidatas", "top-1", "0.8123", "_log_topk_149.txt"),
    ("A · despliegue", "lista de candidatas", "top-3", "0.8663", "_log_topk_149.txt"),

    # ---------------- BASE A: parecido y mundo abierto ----------------
    ("A · parecido", "con hermana contenida >= 0,5", "acierto cascada", "0.9891", "_log_similitud_p2bal_149.txt"),
    ("A · parecido", "sin hermana parecida", "acierto cascada", "0.7434", "_log_similitud_p2bal_149.txt"),
    ("A · parecido", "sin hermana parecida", "acierto texto", "0.5839", "_log_similitud_p2bal_149.txt"),
    ("A · parecido", "tramo alto", "coseno medio con el entrenamiento", "0.8143", "_log_similitud_p2bal_149.txt"),
    ("A · mundo abierto", "familia fuera del catalogo, umbral 0,50", "se abstiene", "0.7909", "_log_mundo_abierto_149.txt"),
    ("A · mundo abierto", "familia conocida, umbral 0,50", "se abstiene", "0.1884", "_log_mundo_abierto_149.txt"),
    ("A · mundo abierto", "familia conocida, umbral 0,50", "acierto en lo que responde", "0.9156", "_log_mundo_abierto_149.txt"),
    ("A · mundo abierto", "con pariente de linaje en el catalogo", "se abstiene", "0.5097", "_log_mundo_abierto_149.txt"),
    ("A · mundo abierto", "sin pariente de linaje", "se abstiene", "0.8421", "_log_mundo_abierto_149.txt"),

    # ---------------- BASE A: que explica el rendimiento ----------------
    ("A · explicativas", "patron interno vs acierto, texto", "Pearson r", "+0.834", "_log_patron_vs_acierto.txt"),
    ("A · explicativas", "patron interno vs acierto, cascada", "Pearson r", "+0.715", "_log_patron_vs_acierto.txt"),
    ("A · explicativas", "ano de aparicion vs F1", "Pearson r", "+0.091", "_log_anio_vs_rendimiento.txt"),

    # ---------------- BASE A: linaje ----------------
    ("A · linaje", "errores del texto solo", "total (50 semillas)", "2093", "_log_confusion_linaje_149.txt"),
    ("A · linaje", "errores del texto, de linaje", "cantidad", "486", "_log_confusion_linaje_149.txt"),
    ("A · linaje", "errores de la cascada", "total (50 semillas)", "1398", "_log_confusion_linaje_149.txt"),
    ("A · linaje", "errores de la cascada, de linaje", "cantidad", "186", "_log_confusion_linaje_149.txt"),
    ("A · linaje", "fusionando los 3 pares fuertes", "exactitud (27 clases)", "0.8585", "_log_acierto_linaje_149.txt"),
    ("A · linaje", "control de fusion aleatoria", "exactitud", "0.8132", "_log_acierto_linaje_149.txt"),
    ("A · linaje", "CLOP-RYUK, clasificador dedicado", "acierto del binario", "0.5878", "_log_jerarquica_linaje_149.txt"),
    ("A · linaje", "BLACKBASTA-CONTI, clasificador dedicado", "acierto del binario", "0.9225", "_log_jerarquica_linaje_149.txt"),
    ("A · linaje", "DHARMA-PHOBOS, clasificador dedicado", "acierto del binario", "0.8660", "_log_jerarquica_linaje_149.txt"),

    # ---------------- BASE A: vias descartadas ----------------
    ("A · descartadas", "cascada jerarquica por linaje", "delta exactitud", "-0.0011", "_log_jerarquica_linaje_149.txt"),
    ("A · descartadas", "capa de contencion", "delta macro-F1", "+0.0000", "_log_capa_contencion.txt"),
    ("A · descartadas", "extension de cifrado", "cobertura", "0.0250", "_log_capa_extension.txt"),
    ("A · descartadas", "extension de cifrado", "acierto donde aplica", "1.0000", "_log_capa_extension.txt"),
    ("A · descartadas", "extension, techo oraculo", "delta macro-F1", "0.0038", "_log_capa_extension.txt"),
    ("A · descartadas", "extension con catalogo MISP", "cobertura", "0.0000", "_log_capa_extension_misp.txt"),
    ("A · descartadas", "ensemble, mejor variante en la cascada", "macro-F1", "0.7406", "_log_ensemble_vistas.txt"),

    # ---------------- BASE A: mejora medida y NO aplicada ----------------
    ("A · mejora NO aplicada", "normalizacion de URL", "macro-F1", "0.7492", "_log_filtro_url_nucleo50.txt"),
    ("A · mejora NO aplicada", "normalizacion de URL", "delta macro-F1", "+0.0076", "_log_filtro_url_nucleo50.txt"),
    ("A · mejora NO aplicada", "normalizacion de URL", "acierto de la regla", "0.9977", "_log_filtro_url_nucleo50.txt"),

    # ---------------- BASE B ----------------
    ("B · 106 familias", "global", "macro-F1", "0.6485", "_log_extension_familias.txt"),
    ("B · 106 familias", "global", "exactitud", "0.7529", "_log_extension_familias.txt"),
    ("B · 106 familias", "restringido a las 30 originales", "macro-F1", "0.7419", "_log_extension_familias.txt"),
    ("B · 106 familias", "solo las 76 familias nuevas", "macro-F1", "0.6502", "_log_extension_familias.txt"),
    ("B · 106 familias", "capa de reglas", "cobertura", "0.5119", "_log_extension_familias.txt"),
    ("B · 106 familias", "capa de reglas", "acierto", "0.9427", "_log_extension_familias.txt"),
    ("B · parecido", "con hermana contenida >= 0,5", "acierto", "0.9587", "_log_extendido_parecido.txt"),
    ("B · parecido", "sin hermana parecida", "acierto", "0.6624", "_log_extendido_parecido.txt"),
    ("B · filtro dominio", "filtro por dominio", "macro-F1", "0.6543", "_log_extendido_parecido.txt"),
    ("B · filtro dominio", "filtro por dominio", "acierto de la regla", "0.9706", "_log_extendido_parecido.txt"),

    # ---------------- inventario ----------------
    ("inventario", "familias nuevas con 2+ plantillas", "cantidad", "77", "_log_inventario_fuentes.txt"),
    ("inventario", "grupos de casi-duplicados que cruzan familias", "cantidad", "25", "_log_inventario_fuentes.txt"),
]

# cifras que NO salen de un log del frente de notas: se declaran con su fuente aparte
EXTERNAS = [
    ("azar multiclase con 30 familias", "1/30 = 0,033", "definicion, no medicion"),
    ("Lemmou et al. F = 0,920", "es la tarea BINARIA nombre-de-nota vs nombre-benigno",
     "NO es clasificacion de familia. El tutor conoce el paper."),
    ("ID Ransomware 71,93 %", "41 aciertos sobre 57 notas de 22 familias",
     "7_compartido_carlos/.../Pruebas.xlsx -- NO es el corpus de la tesis"),
    ("corte de la curva de aprendizaje: 3 plantillas", "bajo P2ret y confirmado bajo P2bal",
     "P2ret entrena con 70,23 plantillas por pliegue y P2bal con ~50: al citar, decir cual"),
    ("3,7 plantillas por familia con las nuevas de tria.ge", "1,68 + 60/30 = 3,68",
     "aritmetica: 1,68 sale de la curva bajo P2bal (todo el corpus); 60 plantillas nuevas sobre 30 familias"),
    ("28 de 30 familias con tres plantillas o mas", "corpus + nuevas >= 3 en todas salvo CRYPTOLOCKER (1) y NOTPETYA (2)",
     "suma de las dos primeras columnas de la tabla de conteos de _log_plantillas_nuevas.txt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-fallas", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("  CIFRAS FINALES DEL FRENTE DE NOTAS -- cada una verificada en su archivo de origen")
    print("=" * 96)
    cache, faltan, grupo_ant = {}, [], None
    okn = 0
    for grupo, desc, metrica, valor, fuente in CIFRAS:
        p = RES / fuente
        if fuente not in cache:
            cache[fuente] = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
        t = cache[fuente]
        if t is None:
            estado, falta = "SIN ARCHIVO", True
        else:
            # se busca el valor tal cual, y tambien sin el signo para los deltas
            v = valor.strip("[]")
            falta = not (valor in t or v in t or v.lstrip("+") in t)
            estado = "ok" if not falta else "NO APARECE"
        if falta:
            faltan.append((grupo, desc, metrica, valor, fuente))
        else:
            okn += 1
        if args.solo_fallas and not falta:
            continue
        if grupo != grupo_ant:
            print(f"\n--- {grupo} " + "-" * (90 - len(grupo)))
            grupo_ant = grupo
        marca = "  " if not falta else "!!"
        print(f"{marca} {desc:<42} {metrica:<34} {valor:>18}   [{estado}]")

    print("\n" + "=" * 96)
    print(f"  VERIFICADAS: {okn} de {len(CIFRAS)}")
    if faltan:
        print(f"  ⚠ NO VERIFICAN {len(faltan)}:")
        for g, d, m, v, f in faltan:
            print(f"      {g} · {d} · {m} = {v}   (buscado en {f})")
    else:
        print("  Todas las cifras aparecen en su archivo de origen.")

    print("\n" + "=" * 96)
    print("  CIFRAS QUE NO SALEN DE UNA CORRIDA -- se citan con su fuente y su aclaracion")
    print("=" * 96)
    for que, valor, nota in EXTERNAS:
        print(f"  {que}\n      {valor}\n      -> {nota}")

    print("\n" + "=" * 96)
    print("  RECORDATORIOS AL CITAR")
    print("=" * 96)
    print("  · Base A (149 notas, 30 familias) y Base B (596 notas, 106) NO son comparables.")
    print("  · «plantilla no vista» es segun coseno de caracteres 0,90, criterio que NO detecta")
    print("    contencion: con contencion >= 0,8 el corpus pasa de 99 a 81 plantillas.")
    print("  · La normalizacion de URL que da 0,7492 NO esta aplicada en las cifras del capitulo.")
    print("  · Ningun numero se cita sin su metrica y su base.")


if __name__ == "__main__":
    main()
