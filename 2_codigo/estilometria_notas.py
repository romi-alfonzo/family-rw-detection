#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estilometria_notas.py -- M.8: ¿la FORMA DE ESCRIBIR identifica a la familia?

LA PREGUNTA (de Romina, 2026-09-09). El frente de notas probó tres espacios de
representación y ninguno mira el estilo:
  · superficie  -> TF-IDF de palabras y de caracteres (`clasificador_notas_v2.py`)
  · significado -> embeddings multilingües (Exp. 3e: (a) empeora, (b) plano)
  · marcadores  -> IOCs y nombre de archivo (M.1, M.6: los dos adoptados)
**La ESTILOMETRÍA nunca se midió** (verificado: cero menciones en ESTADO_TESIS.md,
PLAN_MEJORAS.md, EXPERIMENTOS_PENDIENTES.md y todo `2_codigo/`).

La intuición es buena y vale enunciarla: **la nota la escribe el mismo grupo**, así que su
huella de autor podría persistir entre campañas aunque el contenido cambie — y eso es
exactamente el problema de P2 (reconocer una plantilla no vista de una familia conocida).
El par de HELLOKITTY lo ilustra: `read_me_unlock.txt` («Your have been EPICALLY pwned!!») y
`read_me_ldk.txt` («Hello dear user.») dicen cosas distintas, pero las dos abren con «Hello» y
las dos tienen errores de no-nativo («Your have been», «how i can pay you»).

QUÉ SE MIDE. Rasgos **independientes del contenido**: nada de qué dice la nota, solo cómo está
escrita. Frecuencia de palabras función, de signos de puntuación, uso de mayúsculas, longitudes
de línea/palabra/oración, proporciones de clases de carácter, y rachas repetidas (`!!!!!!!!`,
`--------`). Sin ninguna palabra de contenido, sin IOCs y sin el nombre de la familia.

⚠️ PREDICCIÓN PREREGISTRADA (escrita antes de correr, y es pesimista a propósito):
**(1) Solo estilo va a quedar MUY por encima del azar (0,033) pero MUY por debajo de TF-IDF.**
    Razón: son ~40 features contra ~10⁴, y hay señal de formato obvia (DHARMA usa `.hta` con
    HTML, WASTEDLOCKER son 250 caracteres en mayúsculas).
**(2) Concatenado con la vista canónica NO va a mejorar de forma significativa** (IC 95 % del Δ
    incluye el cero). Razón principal, y hay que declararla: **`char_wb` 3-5 ya captura buena
    parte del estilo** — ve la puntuación, las mayúsculas y las erratas como n-gramas. La
    estilometría explícita sería en gran medida redundante.
**(3) HELLOKITTY no se recupera.** Sus 3 notas son de campañas distintas y una de ellas
    (`read_me_unlock`) es de caza mayor con registro agresivo, mientras la otra es un Q&A
    cortés. Si el estilo la salvara, subiría; predicción: no sube significativamente.
**(4) Límite estructural a declarar:** con 149 notas de mediana ~1.000 caracteres —y una de
    **85**— la estilometría trabaja al borde de lo viable. La literatura de atribución de autoría
    suele pedir miles de palabras por muestra.
**Si (1) y (2) se cumplen, este es el OCTAVO negativo de método convergente**, y refuerza lo
mismo que los otros siete: el techo lo pone el dato.

PROTOCOLO idéntico al canónico: P2 (`grupos`, StratifiedGroupKFold 2 pliegues), mismo
agrupamiento de casi-duplicados, LinearSVC(C=1, class_weight=balanced), Δ pareado por semilla.
Concatenación con **L2 por bloque** (la corrección que Exp. 3e dejó documentada).

Uso:  python estilometria_notas.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
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

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, UMBRAL_NEARDUP, agrupar_neardups,
                                   cargar_corpus, obtener_modelos, vectorizador)

OUT_DEF = _AQUI.parent / "4_resultados" / "resultados_estilometria"

# Palabras funcion: gramaticales, sin carga de contenido. Se usan en atribucion de autoria
# porque el autor las emplea de forma inconsciente y estable.
FUNCION = ["the", "of", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for",
           "on", "are", "with", "as", "i", "his", "they", "be", "at", "one", "have", "this",
           "from", "or", "had", "by", "not", "but", "what", "all", "were", "we", "when",
           "your", "can", "said", "there", "use", "an", "each", "which", "do", "if", "will",
           "if", "no", "any", "our", "us", "me", "my", "so", "then", "them", "these"]
PUNTUACION = list("!?.,:;-_*#|/\\()[]{}\"'<>@+=$%&~`^")


def rasgos(t: str) -> dict:
    """Rasgos de ESTILO, independientes del contenido. Ninguno mira qué dice la nota."""
    n = max(len(t), 1)
    pal = re.findall(r"[A-Za-zÀ-ÿ]+", t)
    npal = max(len(pal), 1)
    lineas = t.split("\n")
    no_vacias = [l for l in lineas if l.strip()]
    oraciones = [x for x in re.split(r"[.!?]+", t) if x.strip()]
    d = {}
    # palabras funcion (frecuencia relativa)
    bajas = Counter(w.lower() for w in pal)
    for w in FUNCION:
        d[f"fw_{w}"] = bajas[w] / npal
    # puntuacion por cada 100 caracteres
    for c in PUNTUACION:
        d[f"pt_{c}"] = 100.0 * t.count(c) / n
    # mayusculas
    letras = [c for c in t if c.isalpha()]
    d["may_ratio"] = sum(1 for c in letras if c.isupper()) / max(len(letras), 1)
    d["pal_TODO_MAY"] = sum(1 for w in pal if len(w) > 1 and w.isupper()) / npal
    d["pal_Capitalizada"] = sum(1 for w in pal if w[:1].isupper() and not w.isupper()) / npal
    # longitudes y estructura
    d["largo_pal_medio"] = float(np.mean([len(w) for w in pal])) if pal else 0.0
    d["largo_linea_medio"] = float(np.mean([len(l) for l in no_vacias])) if no_vacias else 0.0
    d["largo_linea_sd"] = float(np.std([len(l) for l in no_vacias])) if len(no_vacias) > 1 else 0.0
    d["largo_oracion_medio"] = float(np.mean([len(o.split()) for o in oraciones])) if oraciones else 0.0
    d["n_lineas_por_1000c"] = 1000.0 * len(no_vacias) / n
    d["ratio_lineas_vacias"] = (len(lineas) - len(no_vacias)) / max(len(lineas), 1)
    # clases de caracter
    d["ratio_digitos"] = sum(1 for c in t if c.isdigit()) / n
    d["ratio_espacios"] = sum(1 for c in t if c.isspace()) / n
    d["ratio_puntuacion"] = sum(1 for c in t if not c.isalnum() and not c.isspace()) / n
    d["ratio_no_ascii"] = sum(1 for c in t if ord(c) > 127) / n
    # rachas repetidas: !!!!!!!! ------- ======= (marca de estilo muy visible)
    rachas = [len(m.group(0)) for m in re.finditer(r"(.)\1{2,}", t)]
    d["racha_max"] = float(max(rachas)) if rachas else 0.0
    d["n_rachas_por_1000c"] = 1000.0 * len(rachas) / n
    # riqueza lexica (type-token ratio), acotada por longitud
    d["ttr"] = len(set(w.lower() for w in pal)) / npal
    return d


def ic95(delta):
    n = len(delta)
    m = float(np.mean(delta))
    s = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    h = t_dist.ppf(0.975, n - 1) * (s / np.sqrt(n)) if n > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  M.8 -- ESTILOMETRIA: ¿la FORMA DE ESCRIBIR identifica la familia?")
    print("=" * 78)
    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    familias = np.unique(y)
    n = len(textos)

    filas_r = [rasgos(t) for t in textos]
    nombres_r = list(filas_r[0].keys())
    E = np.array([[f[k] for k in nombres_r] for f in filas_r], dtype=float)
    # escalado robusto por columna (mediana/IQR) para que ninguna domine por unidades
    med = np.median(E, axis=0)
    iqr = np.percentile(E, 75, axis=0) - np.percentile(E, 25, axis=0)
    iqr[iqr == 0] = 1.0
    E = (E - med) / iqr

    largos = [len(t) for t in textos]
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))}")
    print(f"Rasgos de estilo: {len(nombres_r)} "
          f"({sum(1 for k in nombres_r if k.startswith('fw_'))} palabras funcion, "
          f"{sum(1 for k in nombres_r if k.startswith('pt_'))} de puntuacion, "
          f"{sum(1 for k in nombres_r if not k.startswith(('fw_','pt_')))} estructurales)")
    print(f"Largo de las notas: min {min(largos)} | mediana {int(np.median(largos))} | "
          f"max {max(largos)} caracteres")
    print(f"⚠️ notas de menos de 300 caracteres (estilometria poco viable): "
          f"{sum(1 for L in largos if L < 300)}")
    print(f"Semillas: {args.n_semillas}\n")

    VAR = ["canonica (TF-IDF combinado)", "solo_estilo", "canonica + estilo (L2 por bloque)"]
    pred = {v: [] for v in VAR}
    for s in range(args.n_semillas):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        yp = {v: np.empty_like(y) for v in VAR}
        for tr, te in cv.split(textos_arr, y, groups=grupos):
            vec = vectorizador("combinado")
            Xtr_t = vec.fit_transform(textos_arr[tr])
            Xte_t = vec.transform(textos_arr[te])
            Etr, Ete = csr_matrix(E[tr]), csr_matrix(E[te])
            bloques = {
                VAR[0]: (Xtr_t, Xte_t),
                VAR[1]: (Etr, Ete),
                VAR[2]: (hstack([normalize(Xtr_t), normalize(Etr)]),
                         hstack([normalize(Xte_t), normalize(Ete)])),
            }
            for v in VAR:
                Xa, Xb = bloques[v]
                clf = obtener_modelos(s)["LinearSVC"]
                clf.fit(Xa, y[tr])
                yp[v][te] = clf.predict(Xb)
        for v in VAR:
            pred[v].append(yp[v])
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas")

    def met(ps):
        return (np.array([f1_score(y, p, average="macro", zero_division=0) for p in ps]),
                np.array([accuracy_score(y, p) for p in ps]),
                np.array([balanced_accuracy_score(y, p) for p in ps]),
                np.array([precision_recall_fscore_support(
                    y, p, labels=familias, zero_division=0)[2] for p in ps]))

    f1_b, acc_b, bal_b, fam_b = met(pred[VAR[0]])
    filas, filas_fam = [], []
    AZAR = 0.033
    for v in VAR:
        f1_v, acc_v, bal_v, fam_v = met(pred[v])
        d = f1_v - f1_b
        m, lo, hi = ic95(d)
        filas.append(dict(vista=v, f1_macro=round(float(f1_v.mean()), 4),
                          f1_macro_sd=round(float(f1_v.std(ddof=1)), 4),
                          veces_el_azar=round(float(f1_v.mean()) / AZAR, 1),
                          exactitud=round(float(acc_v.mean()), 4),
                          exactitud_balanceada=round(float(bal_v.mean()), 4),
                          delta_vs_canonica=round(m, 4) if v != VAR[0] else np.nan,
                          ic95_bajo=round(lo, 4) if v != VAR[0] else np.nan,
                          ic95_alto=round(hi, 4) if v != VAR[0] else np.nan,
                          semillas_positivas=f"{int((d>0).sum())}/{len(d)}" if v != VAR[0] else "",
                          significativo=("SI" if (lo > 0 or hi < 0) else "NO") if v != VAR[0] else ""))
        if v != VAR[0]:
            for j, fam in enumerate(familias):
                mf, lof, hif = ic95(fam_v[:, j] - fam_b[:, j])
                filas_fam.append(dict(vista=v, familia=fam,
                                      f1_canonica=round(float(fam_b[:, j].mean()), 4),
                                      f1_vista=round(float(fam_v[:, j].mean()), 4),
                                      delta=round(mf, 4), ic95_bajo=round(lof, 4),
                                      ic95_alto=round(hif, 4)))

    pd.DataFrame(filas).to_csv(OUT / "m8_resumen.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_fam).to_csv(OUT / "m8_por_familia.csv", index=False, encoding="utf-8-sig")

    print("\n=== RESULTADO ===")
    for r in filas:
        base = r["vista"] == VAR[0]
        print(f"  {r['vista']:<36} macro-F1 {r['f1_macro']:.4f} ± {r['f1_macro_sd']:.4f}"
              f"  ({r['veces_el_azar']}x azar)"
              + ("" if base else f" | Δ {r['delta_vs_canonica']:+.4f} "
                                f"[{r['ic95_bajo']:+.4f}; {r['ic95_alto']:+.4f}] "
                                f"{r['semillas_positivas']} signif:{r['significativo']}"))

    print("\n=== CONTROL DE LAS PREDICCIONES PREREGISTRADAS ===")
    solo = next(r for r in filas if r["vista"] == VAR[1])
    conc = next(r for r in filas if r["vista"] == VAR[2])
    can = next(r for r in filas if r["vista"] == VAR[0])
    p1 = solo["f1_macro"] > AZAR * 3 and solo["f1_macro"] < can["f1_macro"]
    p2 = conc["ic95_bajo"] <= 0 <= conc["ic95_alto"]
    print(f"  (1) solo estilo por encima del azar pero por debajo de TF-IDF: "
          f"{'SE CUMPLE' if p1 else 'NO SE CUMPLE'} "
          f"({solo['f1_macro']:.4f} vs azar {AZAR} y canonica {can['f1_macro']:.4f})")
    print(f"  (2) concatenado NO mejora significativamente: "
          f"{'SE CUMPLE' if p2 else 'NO SE CUMPLE'} "
          f"(Δ {conc['delta_vs_canonica']:+.4f}, IC [{conc['ic95_bajo']:+.4f}; {conc['ic95_alto']:+.4f}])")
    ff = pd.DataFrame(filas_fam)
    hk = ff[(ff.vista == VAR[2]) & (ff.familia == "HELLOKITTY")]
    if len(hk):
        r = hk.iloc[0]
        p3 = not (r.ic95_bajo > 0)
        print(f"  (3) HELLOKITTY no se recupera: {'SE CUMPLE' if p3 else 'NO SE CUMPLE'} "
              f"({r.f1_canonica:.4f} -> {r.f1_vista:.4f}, Δ {r.delta:+.4f} "
              f"[{r.ic95_bajo:+.4f}; {r.ic95_alto:+.4f}])")

    print("\n=== familias que MAS suben con estilo concatenado ===")
    top = ff[ff.vista == VAR[2]].nlargest(6, "delta")
    for _, r in top.iterrows():
        sig = "signif." if r.ic95_bajo > 0 else "n.s."
        print(f"   {r.familia:<14} {r.f1_canonica:.4f} -> {r.f1_vista:.4f}  "
              f"Δ {r.delta:+.4f} [{r.ic95_bajo:+.4f}; {r.ic95_alto:+.4f}]  {sig}")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
