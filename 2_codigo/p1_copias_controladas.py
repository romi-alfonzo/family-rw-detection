#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_copias_controladas.py -- la cascada bajo P1 con el catálogo ampliado por copias controladas.

POR QUÉ. Romina (2026-10-01): «ampliá el corpus de forma controlada para mejorar los datos» y
«probá con duplicados controlados». mejoras_cascada_p1.py mostró que la técnica está en su techo
bajo P1 (V1 +0,0000; V2 +0,0030): el error está en las 76 notas que son el único ejemplar de su
plantilla. Este experimento mide qué pasa si el catálogo tiene más ejemplares reales.

COPIA CONTROLADA. Otra nota real de la MISMA plantilla (otra víctima u otra variante de la misma
familia), con procedencia. La define inventario_copias_controladas.py, que solo lee las fuentes
locales:
  - coseno char 3-5 > 0,90 (TFIDF_CHAR, el umbral que define «plantilla») con alguna nota del corpus;
  - todas las notas del corpus por encima de 0,90 son de la familia de la carpeta y de UNA plantilla;
  - texto no idéntico a ninguna nota del corpus ni a otra candidata;
  - se EXCLUYE si sus marcadores son iguales a los de su vecina, porque puede ser el mismo documento
    transcripto de nuevo (pcrisk_maze_attention_base64.txt frente a pcrisk_maze_1.txt).
Inventario del 2026-10-02: las fuentes que el corpus ya cita (Lemmou, ThreatLabz, PCrisk y la
recolección de agosto) tienen todas sus notas de las 30 familias en el corpus, salvo 7 copias; con
la regla de marcadores quedan 6: CERBER 1 (Lemmou), CHIMERA 1 (Malwarebytes, hasherezade, OCR),
MAZE 1 (PCrisk) y MEDUZALOCKER 3 (PCrisk). f6dfir no tiene ninguna de las 30 familias. Las notas de
otras familias que copian una plantilla del corpus (Monti y Zeon de CONTI, BTCWare de DHARMA,
Lapiovra y Hades de SODINOKIBI) NO se usan: cambiarían la etiqueta.

PROTOCOLO. P1 exacto: StratifiedKFold de 2 pliegues con mezcla, random_state = s, 50 semillas. La
PRUEBA son siempre las 149 notas del corpus, partidas igual que en P1 publicado, así que la
comparación es pareada y sobre la misma base (149 notas, 30 familias).
  V0  cascada publicada: reglas -> texto, entrenada con el pliegue de entrenamiento.
  C1  la misma cascada, con las 6 copias AGREGADAS al entrenamiento de cada pliegue: entran al
      ajuste del TF-IDF, al LinearSVC y al diccionario de marcadores privados. No aportan nombre de
      archivo (no está auditado), así que la regla por nombre no cambia. Nunca se prueban.
Exactitud y macro-F1 (labels = las 30 familias); diferencias pareadas C1 - V0 con IC t sobre 50
semillas; decisiones que cambian; desglose entre las notas cuya plantilla recibe una copia y las
demás; porcentaje de decisiones con alguna hermana de su plantilla en entrenamiento.

PUERTA. V0 tiene que reproducir la cascada publicada bajo P1: 0,8866 / 0,8592. Si no, se aborta.

PREREGISTRO (commiteado ANTES de correr, 2026-10-02).
  CC-0  El grupo controlado tiene 6 copias y da hermana a 3 notas hoy únicas (CHIMERA
        pcrisk_chimera_ingles_autentico, MEDUZALOCKER chip y rapid). Las decisiones con hermana en
        entrenamiento pasan del 41,4 % a entre 43 % y 45 %.
  CC-1  C1 - V0 en exactitud: entre 0 y +0,010. Por construcción no puede pasar de unos +0,03 si
        no cambia nada fuera de las plantillas ampliadas.
  CC-2  C1 - V0 en macro-F1: entre 0 y +0,02.
  CC-3  En las notas cuya plantilla NO recibe copia, la exactitud cambia menos de 0,003 en valor
        absoluto: solo cambian por el reajuste del TF-IDF y del SVM.
LO QUE HAY QUE DECIR PASE LO QUE PASE. Con las fuentes locales, la ampliación controlada da 6 copias
para 76 notas únicas: 73 siguen sin hermana. Mover P1 de verdad pide conseguir otras víctimas de
esas plantillas, y eso es recolectar en fuentes nuevas: es decisión de Romina.
"""
from __future__ import annotations

import csv
import os
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402
from extractor_notas import extraer_texto  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
TOL = 0.0005
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "3_datos"
SALIDA = Path(os.environ.get("SALIDA_COPIAS", RAIZ / "4_resultados" / "resultados_copias_controladas"))  # 2026-10-02: configurable para correr otros grupos sin pisar este
INVENTARIO = SALIDA / "inventario.csv"
VARIANTES = ("V0", "C1")

# --- corpus, igual que mejoras_cascada_p1.py -------------------------------------------------
textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y, grupos = np.asarray(y), np.asarray(grupos)
familias = np.unique(y)
n = len(y)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
tam = Counter(grupos)

# --- copias controladas ----------------------------------------------------------------------
if not INVENTARIO.is_file():
    sys.exit(f"Falta {INVENTARIO}: correr antes inventario_copias_controladas.py")
with open(INVENTARIO, encoding="utf-8") as fh:
    filas = [r for r in csv.DictReader(fh) if r["estado"] == "COPIA" and r["marcadores_iguales"] != "si"]
c_textos, c_y, c_iocs, c_grupo = [], [], [], []
print(f"Copias controladas: {len(filas)}")
for r in filas:
    p = DATOS / r["ruta"]
    texto, metodo = extraer_texto(p)
    j = archivos.index(r["vecina"])
    c_textos.append(texto)
    c_y.append(r["familia"])
    c_iocs.append(set(ab.extraer_marcadores(texto)))
    c_grupo.append(grupos[j])
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"  {r['familia']:13s} cos {r['coseno_max']}  sha256 {sha}  {r['ruta']}  ~ {r['vecina']}"
          f" (plantilla de {tam[grupos[j]]})")
k = len(c_textos)
c_textos_arr = np.array(c_textos, dtype=object)
c_y = np.asarray(c_y)
g_copias = set(c_grupo)
ampliada = np.array([g in g_copias for g in grupos])
unicas_ganan = [archivos[i] for i in range(n) if tam[grupos[i]] == 1 and ampliada[i]]
print(f"Notas del corpus cuya plantilla recibe copia: {ampliada.sum()} | "
      f"únicas que ganan hermana: {len(unicas_ganan)} {unicas_ganan}")

# índices extendidos para el diccionario de marcadores privados
iocs_ext = iocs + c_iocs
nombres_ext = nombres_nota + [None] * k
y_ext = np.concatenate([y, c_y])
idx_copias = list(range(n, n + k))


def regla(i, d):
    claves = set(iocs[i])
    if nombres_nota[i]:
        claves.add(("[NOMBRE]", nombres_nota[i]))
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


pred = {v: np.empty((N_SEM, n), dtype=object) for v in VARIANTES}
capa = {v: np.empty((N_SEM, n), dtype=object) for v in VARIANTES}
hermana = {v: np.zeros((N_SEM, n), dtype=bool) for v in VARIANTES}
for s in range(N_SEM):
    cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
    for tr, te in cv.split(textos_arr, y):
        for v in VARIANTES:
            if v == "V0":
                t_tr, y_tr, tr_ext = textos_arr[tr], y[tr], list(tr)
                g_tr = set(grupos[tr])
            else:
                t_tr = np.concatenate([textos_arr[tr], c_textos_arr])
                y_tr = np.concatenate([y[tr], c_y])
                tr_ext = list(tr) + idx_copias
                g_tr = set(grupos[tr]) | g_copias
            vec = ab.vectorizador("combinado")
            Xtr = vec.fit_transform(t_tr)
            Xte = vec.transform(textos_arr[te])
            clf = ab.obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y_tr)
            top1 = clf.classes_[np.argmax(clf.decision_function(Xte), axis=1)]
            dic = ab.dicc_privados(tr_ext, iocs_ext, nombres_ext, y_ext)
            for kk, i in enumerate(te):
                hermana[v][s, i] = grupos[i] in g_tr
                fam = regla(i, dic)
                if fam is not None:
                    pred[v][s, i], capa[v][s, i] = fam, "regla"
                else:
                    pred[v][s, i], capa[v][s, i] = top1[kk], "texto"
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{N_SEM} semillas", flush=True)

ac = {v: np.array([accuracy_score(y, pred[v][s]) for s in range(N_SEM)]) for v in VARIANTES}
f1 = {v: np.array([f1_score(y, pred[v][s], average="macro", labels=familias, zero_division=0)
                   for s in range(N_SEM)]) for v in VARIANTES}

print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA: V0 = cascada publicada bajo P1\n" + "-" * 78)
ok = abs(ac["V0"].mean() - 0.8866) <= TOL and abs(f1["V0"].mean() - 0.8592) <= TOL
print(f"  V0 exactitud {ac['V0'].mean():.4f} (0,8866) | macro-F1 {f1['V0'].mean():.4f} (0,8592)  "
      f"{'OK' if ok else 'NO REPRODUCE'}")
if not ok:
    sys.exit("ABORTADO: V0 no reproduce la cascada publicada. No se reporta nada.")


def ic(d):
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


print("\n" + "=" * 78 + f"\n  P1, prueba = las 149 notas, 30 familias, {N_SEM} semillas\n" + "=" * 78)
da, la, ha = ic(ac["C1"] - ac["V0"])
dfm, lf, hf = ic(f1["C1"] - f1["V0"])
for v in VARIANTES:
    print(f"  {v:<4} exactitud {ac[v].mean():.4f} ± {ac[v].std(ddof=1):.4f} | "
          f"macro-F1 {f1[v].mean():.4f} ± {f1[v].std(ddof=1):.4f} | "
          f"decisiones con hermana en entrenamiento {hermana[v].mean():.1%}")
print(f"  C1 - V0: exactitud {da:+.4f} [{la:+.4f}; {ha:+.4f}] | macro-F1 {dfm:+.4f} [{lf:+.4f}; {hf:+.4f}]")
mejor = int(((ac["C1"] - ac["V0"]) > 0).sum())
peor = int(((ac["C1"] - ac["V0"]) < 0).sum())
print(f"  semillas: C1 mejor en {mejor}, peor en {peor}, igual en {N_SEM - mejor - peor}")

cambia = pred["C1"] != pred["V0"]
a_bien = cambia & (pred["C1"] == y)
a_mal = cambia & (pred["V0"] == y)
print(f"\n  Decisiones que cambian: {cambia.sum()} | {a_bien.sum()} pasan a acierto, {a_mal.sum()} pasan a error"
      f" | por capa en C1: {dict(Counter(capa['C1'][cambia]))}")
for m, etq in ((ampliada, "plantilla con copia"), (~ampliada, "plantilla sin copia")):
    acv = np.mean([accuracy_score(y[m], pred["V0"][s][m]) for s in range(N_SEM)])
    acc = np.mean([accuracy_score(y[m], pred["C1"][s][m]) for s in range(N_SEM)])
    print(f"  {etq:20s} {m.sum():3d} notas: V0 {acv:.4f} -> C1 {acc:.4f} ({acc - acv:+.4f})")
for i in np.where(ampliada)[0]:
    print(f"    {y[i]:13s} {archivos[i]:55s} V0 {np.mean(pred['V0'][:, i] == y[i]):.2f} -> "
          f"C1 {np.mean(pred['C1'][:, i] == y[i]):.2f}")

SALIDA.mkdir(parents=True, exist_ok=True)
with open(SALIDA / "p1_copias_por_semilla.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["semilla", "exactitud_V0", "exactitud_C1", "macroF1_V0", "macroF1_C1"])
    for s in range(N_SEM):
        w.writerow([s, f"{ac['V0'][s]:.6f}", f"{ac['C1'][s]:.6f}", f"{f1['V0'][s]:.6f}", f"{f1['C1'][s]:.6f}"])

print("\n  VEREDICTO DEL PREREGISTRO")
hc = hermana["C1"].mean()
print(f"  [{'CUMPLE' if k == 6 and len(unicas_ganan) == 3 and 0.43 <= hc <= 0.45 else 'FALLA '}] CC-0 "
      f"6 copias, 3 únicas ganan hermana, con hermana entre 43 % y 45 %   {k} | {len(unicas_ganan)} | {hc:.1%}")
print(f"  [{'CUMPLE' if 0 <= da <= 0.010 else 'FALLA '}] CC-1 delta exactitud entre 0 y +0,010   {da:+.4f}")
print(f"  [{'CUMPLE' if 0 <= dfm <= 0.02 else 'FALLA '}] CC-2 delta macro-F1 entre 0 y +0,02   {dfm:+.4f}")
m = ~ampliada
d_resto = (np.mean([accuracy_score(y[m], pred["C1"][s][m]) for s in range(N_SEM)])
           - np.mean([accuracy_score(y[m], pred["V0"][s][m]) for s in range(N_SEM)]))
print(f"  [{'CUMPLE' if abs(d_resto) < 0.003 else 'FALLA '}] CC-3 sin copia, |delta| < 0,003   {d_resto:+.4f}")
print(f"\nSalidas en {SALIDA}")
