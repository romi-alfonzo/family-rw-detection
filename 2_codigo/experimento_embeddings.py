#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experimento_embeddings.py -- Exp. 3e: representacion semantica multilingue para el
clasificador de notas de rescate.

QUE HACE
Ejecuta el diseno PREREGISTRADO en ESTADO_TESIS.md (bloque "PREDICCIONES PREREGISTRADAS
-- experimento de embeddings multilingues (Exp. 3e)"). NO cambia el metodo de evaluacion:
reusa EXACTAMENTE la particion canonica (StratifiedGroupKFold, N_FOLDS=2, mismas 10
semillas, mismos grupos de casi-duplicados) y el mismo clasificador (LinearSVC(seed)).
Unica variable: la representacion.

BASE A BATIR (P2 sobre 155 notas / 106 plantillas / 30 familias, protocolo "grupos"):
  grupos + combinado + LinearSVC = macro-F1 0,5265 +/- 0,0490
  (4_resultados/resultados_extension_155/resultados_canonicos/corrida_canonica_resumen.csv)
  NOTA: la preregistracion cito "0,530 +/- 0,062"; ese valor es la fila de Regresion
  Logistica (0,5300 +/- 0,0617), modelo distinto. El diseno exige "unica variable = la
  representacion", asi que el modelo se mantiene fijo en LinearSVC (el canonico del frente
  de notas) y la base es 0,5265 +/- 0,0490. El script RECOMPUTA esa base en el mismo proceso
  y aborta si no la reproduce (control de particion identica).

TRES REPRESENTACIONES (todas con el MISMO LinearSVC, protocolo "grupos"):
  base    : combinado = FeatureUnion(word TF-IDF, char TF-IDF)   [reproduce la base 155]
  emb     : embedding multilingue EN LUGAR de TF-IDF             [variante (a)]
  emb+tfidf: embedding CONCATENADO con combinado TF-IDF          [variante (b)]

MODELO DE EMBEDDING (dependencia a declarar en metodologia):
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dim 384), embeddings
  L2-normalizados (normalize_embeddings=True). El modelo es preentrenado y FIJO: no se
  ajusta con los datos, por lo que no hay fuga aunque se calcule sobre todo el corpus.
  Las versiones exactas van al manifiesto.

SALIDA: 4_resultados/resultados_embeddings_155/ (carpeta NUEVA; no toca resultados_canonicos
ni resultados_extension_155).

Uso:
    python experimento_embeddings.py
    python experimento_embeddings.py --salida <carpeta>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy import sparse
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, N_SEMILLAS, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus, obtener_modelos,
                                   vectorizador)

OUT_DIR_DEFAULT = (_AQUI.parent / "4_resultados" / "resultados_embeddings_155"
                   if (_AQUI.parent / "4_resultados").is_dir()
                   else _AQUI / "resultados_embeddings_155")

MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
AGREGACION_EMB = "chunk_mean_pool_l2"  # debe coincidir con construir_embeddings_notas.py
BASE_MACRO_F1 = 0.5265        # grupos+combinado+LinearSVC sobre 155 (control)
BASE_MACRO_F1_STD = 0.0490
TOL_BASE = 0.003              # el control debe reproducir la base dentro de esta tolerancia
FAMILIAS_PREREG = ["JIGSAW", "CHIMERA", "GANDCRAB", "RYUK"]  # 3 suben, RYUK control negativo


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"no serializable: {type(o)}")


def asegurar_embeddings(out_dir, archivos):
    """Carga embeddings.npy (lo genera en un PROCESO APARTE si falta) y verifica que
    cada fila corresponde a la nota del mismo indice de cargar_corpus. torch NO se importa
    en este proceso: el analisis usa solo numpy/scipy/sklearn (camino estable en Windows)."""
    npy = out_dir / "embeddings.npy"
    meta_p = out_dir / "embeddings_meta.json"
    regen = not (npy.exists() and meta_p.exists())
    if not regen:
        meta_prev = json.loads(meta_p.read_text(encoding="utf-8"))
        # Re-generar si el metodo de agregacion cambio (p. ej. se agrego el troceado) o si
        # los embeddings guardados no corresponden al corpus actual: nunca reutilizar viejos.
        if (meta_prev.get("aggregation") != AGREGACION_EMB
                or list(meta_prev.get("archivos", [])) != list(archivos)):
            regen = True
    if regen:
        print("  (embeddings ausentes/desactualizados; generando en proceso aparte con torch)")
        cmd = [sys.executable, str(_AQUI / "construir_embeddings_notas.py"),
               "--salida", str(out_dir)]
        subprocess.run(cmd, check=True)
    emb = np.load(npy)
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if list(meta.get("archivos", [])) != list(archivos):
        sys.exit("ABORTA: los embeddings guardados no estan alineados con el corpus actual "
                 "(lista de archivos distinta). Borrar embeddings.npy y re-generar.")
    if emb.shape[0] != len(archivos):
        sys.exit(f"ABORTA: embeddings {emb.shape[0]} filas != {len(archivos)} notas.")
    if meta.get("aggregation") != AGREGACION_EMB:
        sys.exit(f"ABORTA: agregacion {meta.get('aggregation')} != {AGREGACION_EMB}.")
    versiones = {k: meta[k] for k in ("sentence_transformers", "torch", "transformers",
                                      "dim", "max_seq_length", "normalize_embeddings",
                                      "aggregation", "window_tokens", "stride_tokens",
                                      "n_troceadas", "max_chunks")}
    return emb.astype(np.float32), versiones


def repr_fold(tipo, textos, emb, tr, te):
    """Devuelve (Xtr, Xte) para el fold segun la representacion. El TF-IDF se ajusta SOLO
    con el train (sin fuga, C2). El embedding es fijo (preentrenado) e indexado por fold."""
    if tipo == "emb":
        return emb[tr], emb[te]
    vec = vectorizador("combinado")
    Xtr_t = vec.fit_transform(textos[tr])
    Xte_t = vec.transform(textos[te])
    if tipo == "base":
        return Xtr_t, Xte_t
    if tipo == "emb+tfidf":
        # NORMALIZACION L2 POR BLOQUE antes de concatenar: el bloque TF-IDF combinado son
        # ~10.000 dims ralas (norma ~sqrt(2)) y el embedding 384 densas. Sin igualar la
        # escala, el embedding queda enterrado y no se podria distinguir "no aporta" de
        # "quedo aplastado". Se lleva cada bloque a norma 1 por fila y se concatena.
        Xtr = sparse.hstack([normalize(Xtr_t), sparse.csr_matrix(normalize(emb[tr]))],
                            format="csr")
        Xte = sparse.hstack([normalize(Xte_t), sparse.csr_matrix(normalize(emb[te]))],
                            format="csr")
        return Xtr, Xte
    raise ValueError(tipo)


def evaluar_repr(tipo, textos, y, grupos, familias, emb):
    """Replica EXACTA de la particion de clasificador_notas_v2.evaluar, pero guardando
    macro-F1 POR SEMILLA y F1 POR FAMILIA POR SEMILLA (que evaluar() no expone)."""
    textos = np.array(textos, dtype=object)
    f1_macro_semilla, acc_semilla, balacc_semilla = [], [], []
    f1_fam_semilla = []  # matriz [semilla, familia]
    for seed in range(N_SEMILLAS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        y_pred = np.empty_like(y)
        for tr, te in cv.split(textos, y, groups=grupos):
            clf = obtener_modelos(seed)["LinearSVC"]
            Xtr, Xte = repr_fold(tipo, textos, emb, tr, te)
            clf.fit(Xtr, y[tr])
            y_pred[te] = clf.predict(Xte)
        f1_macro_semilla.append(f1_score(y, y_pred, average="macro", zero_division=0))
        acc_semilla.append(accuracy_score(y, y_pred))
        balacc_semilla.append(balanced_accuracy_score(y, y_pred))
        _, _, f, _ = precision_recall_fscore_support(y, y_pred, labels=familias,
                                                     zero_division=0)
        f1_fam_semilla.append(f)
    return (np.array(f1_macro_semilla), np.array(acc_semilla),
            np.array(balacc_semilla), np.array(f1_fam_semilla))


def ic95_media(delta):
    """IC 95 % de la media de las diferencias pareadas (t de Student, df = n-1)."""
    n = len(delta)
    m = float(np.mean(delta))
    s = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * se if n > 1 else 0.0
    return m, s, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXP. 3e -- REPRESENTACION SEMANTICA MULTILINGUE (notas de rescate)")
    print("=" * 78)
    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    grupos, pares = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Notas: {len(textos)} | Familias: {len(familias)} | "
          f"Plantillas (grupos casi-dup): {len(set(grupos))}")
    print(f"Base a batir (P2 grupos+combinado+LinearSVC, 155): "
          f"macro-F1 {BASE_MACRO_F1:.4f} +/- {BASE_MACRO_F1_STD:.4f}")

    print(f"\nAsegurando embeddings con {MODELO_EMB} ...")
    emb, versiones = asegurar_embeddings(OUT, archivos)
    print(f"  embeddings: {emb.shape} | {versiones}")

    # ---- Evaluar las tres representaciones sobre la MISMA particion
    resultados = {}
    for tipo in ("base", "emb", "emb+tfidf"):
        print(f"\n[{tipo}] evaluando P2 (grupos, LinearSVC, {N_SEMILLAS} semillas) ...",
              end="", flush=True)
        f1m, acc, bal, f1fam = evaluar_repr(tipo, textos, y, grupos, familias, emb)
        resultados[tipo] = dict(f1_macro=f1m, accuracy=acc, balanced_accuracy=bal,
                                f1_fam=f1fam)
        print(f" macro-F1 {f1m.mean():.4f} +/- {f1m.std(ddof=1):.4f}")

    # ---- CONTROL: la base recomputada debe reproducir 0,5265 +/- 0,0490
    base_f1 = resultados["base"]["f1_macro"]
    dif_control = abs(base_f1.mean() - BASE_MACRO_F1)
    print("\n" + "-" * 78)
    print(f"CONTROL DE PARTICION: base recomputada {base_f1.mean():.4f} +/- "
          f"{base_f1.std(ddof=1):.4f} vs almacenada {BASE_MACRO_F1:.4f} +/- "
          f"{BASE_MACRO_F1_STD:.4f} | dif {dif_control:.4f}")
    if dif_control > TOL_BASE:
        sys.exit(f"ABORTA: la base recomputada no reproduce la almacenada "
                 f"(dif {dif_control:.4f} > tol {TOL_BASE}). Particiones distintas: "
                 f"el pareo no seria valido.")
    print("  OK: particion identica, el Delta pareado es valido.")

    idx_fam = {f: i for i, f in enumerate(familias)}

    # ---- (2) macro-F1 global con Delta pareado e IC 95 %
    filas_global = []
    for tipo in ("emb", "emb+tfidf"):
        v = resultados[tipo]["f1_macro"]
        delta = v - base_f1
        m, s, lo, hi = ic95_media(delta)
        filas_global.append(dict(
            representacion=tipo,
            base_mean=round(float(base_f1.mean()), 4),
            base_std=round(float(base_f1.std(ddof=1)), 4),
            var_mean=round(float(v.mean()), 4),
            var_std=round(float(v.std(ddof=1)), 4),
            delta_mean=round(m, 4), delta_std=round(s, 4),
            ic95_inf=round(lo, 4), ic95_sup=round(hi, 4),
            n_semillas_delta_pos=int(np.sum(delta > 0))))
    pd.DataFrame(filas_global).to_csv(OUT / "emb_delta_global.csv", index=False)

    # ---- (1) F1 por familia (media +/- desvio) + Delta pareado, TODAS las familias
    filas_fam = []
    for tipo in ("emb", "emb+tfidf"):
        vfam = resultados[tipo]["f1_fam"]        # [semilla, familia]
        bfam = resultados["base"]["f1_fam"]
        for fam in familias:
            j = idx_fam[fam]
            b = bfam[:, j]; vv = vfam[:, j]; d = vv - b
            m, s, lo, hi = ic95_media(d)
            filas_fam.append(dict(
                representacion=tipo, familia=fam, prereg=fam in FAMILIAS_PREREG,
                base_mean=round(float(b.mean()), 4), base_std=round(float(b.std(ddof=1)), 4),
                var_mean=round(float(vv.mean()), 4), var_std=round(float(vv.std(ddof=1)), 4),
                delta_mean=round(m, 4), delta_std=round(s, 4),
                ic95_inf=round(lo, 4), ic95_sup=round(hi, 4),
                n_semillas_delta_pos=int(np.sum(d > 0))))
    df_fam = pd.DataFrame(filas_fam)
    df_fam.to_csv(OUT / "emb_por_familia.csv", index=False)

    # ---- por-semilla crudo (trazabilidad del pareo)
    filas_sem = []
    for tipo in ("base", "emb", "emb+tfidf"):
        for s_i in range(N_SEMILLAS):
            filas_sem.append(dict(
                representacion=tipo, semilla=s_i,
                f1_macro=round(float(resultados[tipo]["f1_macro"][s_i]), 6),
                accuracy=round(float(resultados[tipo]["accuracy"][s_i]), 6),
                balanced_accuracy=round(float(resultados[tipo]["balanced_accuracy"][s_i]), 6)))
    pd.DataFrame(filas_sem).to_csv(OUT / "emb_por_semilla.csv", index=False)

    # ---- Reporte en pantalla, en el orden pedido
    print("\n" + "=" * 78)
    print("  (1) F1 POR FAMILIA -- familias preregistradas (media +/- desvio, 10 semillas)")
    print("=" * 78)
    for tipo in ("emb", "emb+tfidf"):
        print(f"\n  [{tipo}]  (Delta = variante - base, pareado por semilla)")
        print(f"    {'familia':12} {'base':>16} {'variante':>16} {'Delta':>9} "
              f"{'IC95':>20} {'sem+':>5}")
        sub = df_fam[df_fam.representacion == tipo].set_index("familia")
        for fam in FAMILIAS_PREREG:
            r = sub.loc[fam]
            print(f"    {fam:12} {r.base_mean:6.3f}+/-{r.base_std:5.3f} "
                  f"{r.var_mean:6.3f}+/-{r.var_std:5.3f} {r.delta_mean:+8.3f} "
                  f"[{r.ic95_inf:+.3f},{r.ic95_sup:+.3f}] {int(r.n_semillas_delta_pos):>3}/10")

    print("\n" + "=" * 78)
    print("  (2) macro-F1 GLOBAL -- Delta pareado e IC 95 %")
    print("=" * 78)
    for r in filas_global:
        print(f"  [{r['representacion']:9}] base {r['base_mean']:.4f}+/-{r['base_std']:.4f}"
              f" -> {r['var_mean']:.4f}+/-{r['var_std']:.4f} | "
              f"Delta {r['delta_mean']:+.4f} IC95 [{r['ic95_inf']:+.4f},{r['ic95_sup']:+.4f}]"
              f" | sem+ {r['n_semillas_delta_pos']}/10")

    # ---- (3) Veredicto contra el criterio de adopcion preregistrado, por variante
    print("\n" + "=" * 78)
    print("  (3) VEREDICTO CONTRA EL CRITERIO DE ADOPCION (por variante)")
    print("=" * 78)
    veredictos = {}
    for tipo in ("emb", "emb+tfidf"):
        sub = df_fam[df_fam.representacion == tipo].set_index("familia")
        suben3 = {f: float(sub.loc[f].delta_mean) for f in ["JIGSAW", "CHIMERA", "GANDCRAB"]}
        ryuk = float(sub.loc["RYUK"].delta_mean)
        g = [r for r in filas_global if r["representacion"] == tipo][0]
        a = all(d > 0 for d in suben3.values())
        # RYUK "no sube de forma comparable": no positivo, o mucho menor que la menor de las 3
        min_sube = min(suben3.values())
        ryuk_comparable = ryuk > 0 and ryuk >= 0.5 * min_sube if min_sube > 0 else ryuk > 0
        c = g["ic95_sup"] >= 0    # el global no empeora: IC95 no enteramente negativo
        if a and not ryuk_comparable and c:
            vd = "ADOPTAR: (a) las 3 suben, (b) RYUK no comparable, (c) global no empeora"
        elif a and ryuk_comparable:
            vd = ("MEJORA GENERICA (mecanismo NO probado): suben las 3 pero TAMBIEN RYUK "
                  "de forma comparable -> se reporta con esa lectura")
        elif not a:
            vd = ("NO ADOPTAR: no suben las 3 multi-idioma -> la variacion entre plantillas "
                  "es de contenido, no de superficie (techo del corpus confirmado)")
        else:
            vd = "NO ADOPTAR: global empeora (IC95 del Delta enteramente negativo)"
        veredictos[tipo] = dict(suben_las_3=a, deltas_3=suben3, delta_ryuk=ryuk,
                                ryuk_comparable=ryuk_comparable,
                                global_no_empeora=c, ic95_global=[g["ic95_inf"], g["ic95_sup"]],
                                veredicto=vd)
        print(f"\n  [{tipo}]")
        print(f"    (a) suben las 3 (JIGSAW/CHIMERA/GANDCRAB): {a}  Delta={suben3}")
        print(f"    (b) RYUK Delta={ryuk:+.4f} -> comparable a las 3? {ryuk_comparable}")
        print(f"    (c) global no empeora (IC95 sup>=0): {c}  IC95=[{g['ic95_inf']:+.4f},{g['ic95_sup']:+.4f}]")
        print(f"    => {vd}")

    # ---- Manifiesto
    (OUT / "manifiesto_emb.json").write_text(json.dumps(dict(
        fecha=str(date.today()), experimento="3e", corpus=str(CORPUS_DIR),
        n_notas=len(textos), n_familias=int(len(familias)),
        n_plantillas=int(len(set(grupos))), pares_neardup=len(pares),
        umbral_neardup=UMBRAL_NEARDUP, n_folds=N_FOLDS, n_semillas=N_SEMILLAS,
        protocolo="grupos (P2)", modelo_clasificador="LinearSVC(C=1.0, class_weight=balanced)",
        modelo_embedding=MODELO_EMB, embedding=versiones,
        representaciones=dict(
            base="combinado = FeatureUnion(word TF-IDF, char TF-IDF), ajustado por pliegue",
            emb="embedding multilingue troceado (mean pooling), L2, EN LUGAR de TF-IDF",
            emb_tfidf="L2 por bloque (TF-IDF combinado y embedding) y luego concatenacion"),
        base_a_batir=dict(fuente="resultados_extension_155/corrida_canonica_resumen.csv",
                          config="grupos+combinado+LinearSVC",
                          macro_f1=BASE_MACRO_F1, macro_f1_std=BASE_MACRO_F1_STD,
                          nota_preregistracion="0,530+/-0,062 = fila Regresion Logistica; "
                                               "el diseno fija LinearSVC (unica variable=repr.)"),
        control_particion=dict(base_recomputada=round(float(base_f1.mean()), 4),
                               dif=round(float(dif_control), 4), tol=TOL_BASE, ok=True),
        familias_preregistradas=FAMILIAS_PREREG,
        veredictos=veredictos,
        sklearn=sklearn.__version__, scipy=scipy.__version__,
        python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"\nSalidas en: {OUT}")


if __name__ == "__main__":
    main()
