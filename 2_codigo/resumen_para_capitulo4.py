#!/usr/bin/env python3.11
"""Agrega los resultados multisemilla y emite las tablas del capítulo 4.

Los experimentos escriben UN archivo por semilla. Este script hace el paso que
faltaba: promediar entre semillas y contar familias. Antes se hacía a mano en el
chat, que es exactamente donde no debe estar — una cifra de la tesis tiene que
poder regenerarse con un comando.

Uso:
    python resumen_para_capitulo4.py                    # usa el job más reciente
    python resumen_para_capitulo4.py --job-estructural 3651
    python resumen_para_capitulo4.py --solo b1          # solo la curva de notas

Salidas en 4_resultados/resumen_capitulo4/:
    2b_ablacion.csv        exactitud/cobertura por modalidad, media ± desvío
    2b_tipos_de_marca.csv  cuántas familias tienen firma, extensión, ambas o nada
    2c_por_familia.csv     F1 por familia, media ± desvío
    b1_curva.csv           B.1: macro-F1 por punto de la curva, media ± desvío
    b1_deltas_pareados.csv B.1: cuánto aporta cada plantilla/nota extra, con IC 95 %
    b1_extrapolacion.csv   B.1: cuántas plantillas por familia harían falta por objetivo
    b1_costo_recoleccion.csv  B.1: cuántas notas nuevas hay que conseguir (aritmética)
    fig_b1_curva_notas.png B.1: la figura del capítulo
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
# En el cluster los archivos van todos planos en /scratch/ralfonzo/tesis, sin la
# estructura del repositorio: hay que caer a la carpeta del script.
RES = (_AQUI.parent / "4_resultados" if (_AQUI.parent / "4_resultados").is_dir()
       else _AQUI)
OUT = RES / "resumen_capitulo4"
MIN_MARCA = 4


def carpetas_estructural(job=None):
    """Carpetas de una MISMA corrida. Filtrar por job es imprescindible: el
    directorio mezcla corridas sueltas (semilla 42, pruebas previas) con la
    tanda buena, y promediarlas juntas falsea el desvío."""
    todas = sorted(RES.glob("**/resultados_estructural_s*_job*"))
    if not todas:
        return [], None
    jobs = {int(re.search(r"_job(\d+)$", str(d)).group(1)) for d in todas}
    job = job or max(jobs)
    return [d for d in todas if d.name.endswith(f"_job{job}")], job


def resumen_2b(job=None):
    dirs, job = carpetas_estructural(job)
    if not dirs:
        print("  (sin carpetas del detector estructural)")
        return
    filas_abl, tipos, semillas = [], [], []
    for d in sorted(dirs):
        man = json.loads((d / "manifiesto.json").read_text(encoding="utf-8"))
        semillas.append(man["semilla"])
        for etiqueta, crit in man["criterios"].items():
            for modo, v in crit["ablacion"].items():
                cubiertos = v["total"] - v["sin_marca"]
                filas_abl.append(dict(
                    criterio=etiqueta, modo=modo, semilla=man["semilla"],
                    exactitud=v["exactitud"], cobertura=v["cobertura"],
                    donde_aplica=round(v["aciertos"] / cubiertos, 4) if cubiertos else 0.0,
                    familias_con_marca=crit["familias_con_marca"]))
        c = Counter()
        for r in csv.DictReader(open(d / "marcas_por_familia_umbral_90.csv", encoding="utf-8")):
            firma = max(int(r["prefijo_len"]), int(r["sufijo_len"])) >= MIN_MARCA
            ext = bool(r["extension"].strip())
            c[("firma y extensión" if firma and ext else "solo firma" if firma
               else "solo extensión" if ext else "sin marca")] += 1
        tipos.append((man["semilla"], c))

    print(f"  job {job} | semillas {sorted(semillas)} | n = {len(semillas)}\n")
    agg = defaultdict(lambda: defaultdict(list))
    for f in filas_abl:
        for m in ("exactitud", "cobertura", "donde_aplica"):
            agg[(f["criterio"], f["modo"])][m].append(f[m])

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "2b_ablacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["criterio", "modo", "exactitud_media", "exactitud_desvio",
                    "cobertura_media", "cobertura_desvio",
                    "donde_aplica_media", "donde_aplica_desvio", "n_semillas"])
        for (crit, modo), d in sorted(agg.items()):
            vals = [x for m in ("exactitud", "cobertura", "donde_aplica")
                    for x in (st.mean(d[m]), st.stdev(d[m]) if len(d[m]) > 1 else 0.0)]
            w.writerow([crit, modo] + [round(v, 4) for v in vals] + [len(d["exactitud"])])
            print(f"  {crit:<12} {modo:<22} exactitud {vals[0]:.4f} ± {vals[1]:.4f} | "
                  f"cobertura {vals[2]:.4f} ± {vals[3]:.4f} | "
                  f"donde aplica {vals[4]:.4f} ± {vals[5]:.4f}")

    print("\n  Tipos de marca (familias):")
    claves = ["firma y extensión", "solo firma", "solo extensión", "sin marca"]
    with open(OUT / "2b_tipos_de_marca.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["tipo", "media", "minimo", "maximo"])
        for k in claves:
            v = [c[k] for _, c in tipos]
            w.writerow([k, round(st.mean(v), 2), min(v), max(v)])
            marca = "" if min(v) == max(v) else "   ← varía entre semillas"
            print(f"    {k:<20} {st.mean(v):>5.1f}  (mín {min(v)}, máx {max(v)}){marca}")


def resumen_2c(umbral=0.98):
    f = next(RES.glob("**/bytes_multisemilla_por_familia.csv"), None)
    if f is None:
        print("  (falta bytes_multisemilla_por_familia.csv)")
        return
    d = defaultdict(list)
    for r in csv.DictReader(open(f, encoding="utf-8")):
        d[r["familia"]].append(float(r["f1"]))
    res = {fam: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
           for fam, v in d.items()}
    n_sem = max(v[2] for v in res.values())
    altas = [f for f, (m, _, _) in res.items() if m >= umbral]
    print(f"  {len(res)} familias | {n_sem} semillas")
    print(f"  F1 ≥ {umbral}: {len(altas)} de {len(res)}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "2c_por_familia.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["familia", "f1_media", "f1_desvio", "n_semillas"])
        for fam, (m, s, n) in sorted(res.items(), key=lambda x: x[1][0]):
            w.writerow([fam, round(m, 4), round(s, 4), n])
    print("  Familias por debajo del umbral:")
    for fam, (m, s, _) in sorted(res.items(), key=lambda x: x[1][0]):
        if m < umbral:
            print(f"    {fam:<15} {m:.3f} ± {s:.3f}")


# ============================================================
# B.1 — CURVA DE APRENDIZAJE DEL FRENTE DE NOTAS
# ============================================================
# Nomenclatura, porque en este proyecto se confunde: PLANTILLA = contenido distinto
# (componente de casi-duplicados, coseno char 3-5 > 0,90). El corpus tiene 144 notas
# pero solo 95 plantillas. "unidad = plantillas" mide DIVERSIDAD (textos nuevos, caro);
# "unidad = notas" mide VOLUMEN (notas al azar, que muchas veces repiten un texto ya
# presente, barato). La separación entre las dos curvas al mismo número de notas de
# entrenamiento es, medida, cuánto vale la diversidad.

_CLAVE_B1 = ["curva", "protocolo", "unidad"]
_METRICAS_B1 = ["f1_macro", "exactitud", "exactitud_balanceada", "f1_weighted"]


def _k_num(v):
    """El eje x mezcla enteros con la etiqueta 'todo'. 'todo' va al final."""
    return float("inf") if str(v) == "todo" else int(v)


def _ajuste_potencia(ks, ys):
    """Ajusta macro-F1(k) = F_inf - a * k^(-b) por mínimos cuadrados.

    Para b fijo el modelo es LINEAL en x = k^(-b), así que se recorre una grilla de b
    y se resuelve el resto exactamente. Es determinista y no depende de que converja
    un optimizador, que con 3 puntos es una fuente de sorpresas.
    """
    import numpy as np
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    mejor = None
    for b in np.arange(0.05, 4.0 + 1e-9, 0.05):
        x = ks ** (-b)
        A = np.column_stack([np.ones_like(x), -x])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        sse = float(((A @ coef - ys) ** 2).sum())
        if mejor is None or sse < mejor[0]:
            mejor = (sse, float(coef[0]), float(coef[1]), float(b))
    sse, F_inf, a, b = mejor
    return dict(F_inf=F_inf, a=a, b=b, sse=sse)


def _k_para_objetivo(fit, objetivo):
    """Plantillas por familia que el ajuste predice para alcanzar `objetivo` de macro-F1.

    Devuelve None si el objetivo está por encima del techo estimado (F_inf) o si la
    curva ajustada no es creciente (a <= 0): en ambos casos la respuesta honesta es
    "no se alcanza agregando datos", no un número.
    """
    if fit["a"] <= 0 or objetivo >= fit["F_inf"]:
        return None
    return float((fit["a"] / (fit["F_inf"] - objetivo)) ** (1.0 / fit["b"]))


def resumen_b1(objetivos=(0.50, 0.60, 0.70, 0.80), n_bootstrap=2000, semilla=7,
               curva_dir=None):
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
    except Exception as e:
        print(f"  (B.1 omitido: falta una dependencia — {e})")
        return
    curva_dir = Path(curva_dir) if curva_dir else (RES / "resultados_curva_notas")
    f = curva_dir / "b1_curva_por_repeticion.csv"
    if not f.exists():
        print(f"  (falta {f.name}: correr primero curva_aprendizaje_notas.py)")
        return
    df = pd.read_csv(f)
    df["k_num"] = df["k"].map(_k_num)
    man_p = curva_dir / "manifiesto_b1.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"  corpus: {man.get('n_notas', '?')} notas · {man.get('n_plantillas', '?')} "
          f"plantillas · {man.get('n_familias', '?')} familias | "
          f"R retención = {man.get('repeticiones_retencion', '?')}")
    print("  azar (macro-F1): 30 familias 0,033 · 11 familias 0,091 · 5 familias 0,200")

    # ---- (1) la curva: media ± desvío por punto
    agg = {m: ["mean", "std"] for m in _METRICAS_B1}
    agg.update({c: "mean" for c in ("n_notas_train", "n_plantillas_train",
                                    "n_fam_bajo_tope", "n_fam_sin_train",
                                    "n_notas_evaluadas", "n_familias")})
    agg["repeticion"] = "count"
    tab = df.groupby(_CLAVE_B1 + ["k", "k_num"]).agg(agg)
    tab.columns = ["_".join(c).rstrip("_") for c in tab.columns]
    tab = tab.reset_index().sort_values(_CLAVE_B1 + ["k_num"])
    tab["notas_train_por_familia"] = tab.n_notas_train_mean / tab.n_familias_mean
    tab["plantillas_train_por_familia"] = (tab.n_plantillas_train_mean
                                           / tab.n_familias_mean)
    tab.drop(columns=["k_num"]).to_csv(OUT / "b1_curva.csv", index=False)

    for (cur, prot, uni), s in tab.groupby(_CLAVE_B1, sort=True):
        print(f"\n  ── {cur} · {prot} · tope por {uni} "
              f"({int(s.n_familias_mean.iloc[0])} familias)")
        print(f"    {'k':>5}  {'macro-F1':>17}  {'exactitud':>17}  "
              f"{'notas/fam':>9}  {'plant./fam':>10}  {'fam.<tope':>9}  {'sin train':>9}")
        for _, r in s.iterrows():
            print(f"    {str(r.k):>5}  {r.f1_macro_mean:.4f} ± {r.f1_macro_std:.4f}  "
                  f"{r.exactitud_mean:.4f} ± {r.exactitud_std:.4f}  "
                  f"{r.notas_train_por_familia:>9.2f}  "
                  f"{r.plantillas_train_por_familia:>10.2f}  "
                  f"{r.n_fam_bajo_tope_mean:>9.1f}  {r.n_fam_sin_train_mean:>9.1f}")

    # ---- (2) deltas PAREADOS entre puntos consecutivos
    # Los puntos comparten la partición y están anidados (el train de k está contenido
    # en el de k+1), así que la diferencia se calcula por repetición. Restar dos medias
    # independientes con un desvío de ±0,05 en macro-F1 no distingue nada.
    filas_d = []
    for (cur, prot, uni), s in df.groupby(_CLAVE_B1, sort=True):
        piv = s.pivot_table(index="repeticion", columns="k", values="f1_macro")
        orden = sorted(piv.columns, key=_k_num)
        for k1, k2 in zip(orden, orden[1:]):
            d = (piv[k2] - piv[k1]).dropna()
            n = len(d)
            if n < 2:
                continue
            ee = d.std(ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            filas_d.append(dict(curva=cur, protocolo=prot, unidad=uni,
                                paso=f"{k1} → {k2}", n=n,
                                delta_f1_macro=d.mean(), desvio=d.std(ddof=1),
                                ic95_inf=d.mean() - t * ee, ic95_sup=d.mean() + t * ee,
                                significativo=bool((d.mean() - t * ee) > 0)))
    dd = pd.DataFrame(filas_d)
    dd.round(4).to_csv(OUT / "b1_deltas_pareados.csv", index=False)
    print("\n  ── Cuánto aporta cada paso (diferencia pareada de macro-F1, IC 95 %)")
    for _, r in dd.iterrows():
        marca = "sí" if r.significativo else "NO — indistinguible de cero"
        print(f"    {r.curva:<6} {r.protocolo:<6} {r.unidad:<10} {r.paso:<12} "
              f"{r.delta_f1_macro:+.4f}  [{r.ic95_inf:+.4f}; {r.ic95_sup:+.4f}]  {marca}")

    # ---- (3) extrapolación, solo donde el conjunto de clases es constante
    filas_e = []
    rng = np.random.default_rng(semilla)
    for (cur, prot, uni), s in df[df.k != "todo"].groupby(_CLAVE_B1, sort=True):
        piv = s.pivot_table(index="repeticion", columns="k", values="f1_macro")
        ks = np.array(sorted(piv.columns, key=_k_num), float)
        if len(ks) < 3:
            continue
        piv = piv[[c for c in sorted(piv.columns, key=_k_num)]]
        fit = _ajuste_potencia(ks, piv.mean(axis=0).values)
        # Bootstrap sobre repeticiones: con 3 puntos y 3 parámetros el ajuste pasa
        # exacto por las medias, así que la única incertidumbre honesta viene de
        # remuestrear las repeticiones.
        boot = {o: [] for o in objetivos}
        techos = []
        for _ in range(n_bootstrap):
            sel = rng.integers(0, len(piv), len(piv))
            fb = _ajuste_potencia(ks, piv.values[sel].mean(axis=0))
            techos.append(fb["F_inf"])
            for o in objetivos:
                kk = _k_para_objetivo(fb, o)
                boot[o].append(np.nan if kk is None else kk)
        for o in objetivos:
            v = np.array(boot[o], float)
            alcanzable = float(np.mean(~np.isnan(v)))
            vv = v[~np.isnan(v)]
            filas_e.append(dict(
                curva=cur, protocolo=prot, unidad=uni, objetivo_f1_macro=o,
                puntos_ajustados=len(ks), techo_estimado=fit["F_inf"],
                techo_ic95_inf=float(np.percentile(techos, 2.5)),
                techo_ic95_sup=float(np.percentile(techos, 97.5)),
                exponente_b=fit["b"], sse=fit["sse"],
                k_estimado=(_k_para_objetivo(fit, o) or float("nan")),
                k_ic95_inf=(float(np.percentile(vv, 2.5)) if len(vv) else float("nan")),
                k_ic95_sup=(float(np.percentile(vv, 97.5)) if len(vv) else float("nan")),
                prob_alcanzable=alcanzable))
    ee_df = pd.DataFrame(filas_e)
    ee_df.round(4).to_csv(OUT / "b1_extrapolacion.csv", index=False)
    print("\n  ── Extrapolación: cuántas unidades por familia harían falta")
    print("     (k_estimado en la MISMA unidad de la fila; 'no alcanzable' = el objetivo")
    print("      está por encima del techo que estima el ajuste)")
    for (cur, prot, uni), s in ee_df.groupby(_CLAVE_B1, sort=True):
        r0 = s.iloc[0]
        print(f"    {cur} · {prot} · tope por {uni} — techo estimado de macro-F1 "
              f"{r0.techo_estimado:.3f} [IC 95 % {r0.techo_ic95_inf:.3f}; "
              f"{r0.techo_ic95_sup:.3f}], {int(r0.puntos_ajustados)} puntos")
        for _, r in s.iterrows():
            if np.isnan(r.k_estimado):
                print(f"        objetivo macro-F1 {r.objetivo_f1_macro:.2f}: "
                      f"NO ALCANZABLE agregando datos "
                      f"(alcanzable en {r.prob_alcanzable * 100:.0f} % del bootstrap)")
            else:
                print(f"        objetivo macro-F1 {r.objetivo_f1_macro:.2f}: "
                      f"{r.k_estimado:.1f} {uni}/familia "
                      f"[IC 95 % {r.k_ic95_inf:.1f}; {r.k_ic95_sup:.1f}] · "
                      f"alcanzable en {r.prob_alcanzable * 100:.0f} % del bootstrap")

    # ---- (4.bis) la lista concreta: qué familia buscar y cuántos textos nuevos
    # Es la salida operativa de B.1: con esto se sale a recolectar. El F1 por familia
    # sale de la MISMA corrida (30fam · P2ret · k=todo), no de la canónica, para que la
    # base sea la misma (144 notas) y el protocolo también.
    ppf = man.get("plantillas_por_familia") or {}
    fpf = curva_dir / "b1_curva_por_familia.csv"
    if ppf and fpf.exists():
        dfam = pd.read_csv(fpf)
        act = dfam[(dfam.curva == "30fam") & (dfam.protocolo == "P2ret")
                   & (dfam.unidad == "plantillas") & (dfam.k.astype(str) == "todo")]
        f1_act = act.groupby("familia").f1.agg(["mean", "std"])
        notas_por_fam = man.get("notas_por_familia") or {}
        filas_r = []
        for fam, n in sorted(ppf.items()):
            filas_r.append(dict(
                familia=fam, plantillas_actuales=n,
                notas_actuales=notas_por_fam.get(fam, ""),
                faltan_para_3=max(0, 3 - n), faltan_para_4=max(0, 4 - n),
                f1_actual=round(float(f1_act["mean"].get(fam, float("nan"))), 4),
                f1_desvio=round(float(f1_act["std"].get(fam, float("nan"))), 4)))
        dr = pd.DataFrame(filas_r).sort_values(
            ["faltan_para_4", "f1_actual"], ascending=[False, True])
        dr.to_csv(OUT / "b1_familias_a_recolectar.csv", index=False)
        print("\n  ── LISTA DE RECOLECCIÓN: qué familia buscar y cuántos textos nuevos")
        print("     (F1 por familia de la MISMA corrida: 30fam · P2ret · k=todo · 144 notas.")
        print("      Es F1 de una familia, NO el macro-F1, que es el promedio de las 30.)")
        print(f"    {'familia':<15} {'plantillas':>10} {'faltan→4':>9} {'faltan→3':>9} "
              f"{'F1 hoy':>16}")
        for _, r in dr.iterrows():
            if r.faltan_para_4 == 0:
                continue
            print(f"    {r.familia:<15} {r.plantillas_actuales:>10} "
                  f"{r.faltan_para_4:>9} {r.faltan_para_3:>9} "
                  f"{r.f1_actual:>8.3f} ± {r.f1_desvio:.3f}")
        print(f"    {'TOTAL':<15} {'':>10} {int(dr.faltan_para_4.sum()):>9} "
              f"{int(dr.faltan_para_3.sum()):>9}")
        completas = dr[dr.faltan_para_4 == 0]
        print(f"    Ya en 4 o más ({len(completas)}): "
              f"{', '.join(completas.familia.tolist())}")

    # ---- (4) cuánto hay que recolectar: aritmética pura, sin modelo
    if ppf:
        filas_c = []
        for k in range(1, 9):
            faltan = {f: max(0, k - n) for f, n in ppf.items()}
            filas_c.append(dict(
                k_plantillas_por_familia=k,
                plantillas_nuevas=sum(faltan.values()),
                familias_a_completar=sum(1 for v in faltan.values() if v > 0),
                familias_ya_completas=sum(1 for v in faltan.values() if v == 0)))
        pd.DataFrame(filas_c).to_csv(OUT / "b1_costo_recoleccion.csv", index=False)
        print("\n  ── Costo de recolección (aritmética sobre el corpus, sin modelo)")
        print("     Cada plantilla nueva exige AL MENOS una nota nueva, y tiene que ser")
        print("     de contenido distinto: una copia de un texto ya presente no cuenta.")
        print(f"    {'k':>3}  {'plantillas nuevas':>18}  {'familias a completar':>20}")
        for r in filas_c:
            print(f"    {r['k_plantillas_por_familia']:>3}  {r['plantillas_nuevas']:>18}  "
                  f"{r['familias_a_completar']:>20}")

    _figura_b1(tab, dd)
    print(f"\n  Salidas de B.1 en: {OUT}")


def _figura_b1(tab, deltas):
    """Tres paneles. (a) y (b) son el MISMO dato sobre dos ejes distintos, y ahi esta el
    resultado: contra notas las dos formas de recolectar se separan, contra plantillas se
    superponen => lo que manda es la cantidad de TEXTOS DISTINTOS, no de notas.
    (c) la curva por plantillas para los tres conjuntos de familias."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (figura de B.1 omitida: {e})")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    sub = tab[(tab.curva == "30fam") & (tab.protocolo == "P2ret") & (tab.k != "todo")]
    series = (("plantillas", "recolectar TEXTOS DISTINTOS (diversidad)", "tab:blue", "o"),
              ("notas", "recolectar notas al azar (volumen)", "tab:orange", "s"))
    for ax, eje, etq_eje in ((ax1, "notas_train_por_familia",
                              "Notas de entrenamiento por familia"),
                             (ax2, "plantillas_train_por_familia",
                              "Plantillas (textos distintos) de entrenamiento por familia")):
        for uni, etq, col, mk in series:
            s = sub[sub.unidad == uni].sort_values(eje)
            ax.errorbar(s[eje], s.f1_macro_mean, yerr=s.f1_macro_std,
                        marker=mk, color=col, capsize=3, label=etq)
        ax.axhline(1 / 30, ls=":", c="grey", lw=1)
        ax.set_xlabel(etq_eje)
        ax.set_ylabel("macro-F1 (P2 con plantilla retenida)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.50, 0.66)
    ax1.set_title("(a) Sobre el eje de NOTAS: las dos curvas se separan\n"
                  "la misma nota no vale lo mismo segun si repite un texto")
    ax2.set_title("(b) Sobre el eje de PLANTILLAS: se superponen\n"
                  "el macro-F1 depende de los textos distintos, no de las notas")

    for cur, col, mk in (("30fam", "tab:blue", "o"), ("11fam", "tab:green", "^"),
                         ("5fam", "tab:red", "v")):
        s = tab[(tab.curva == cur) & (tab.protocolo == "P2ret")
                & (tab.unidad == "plantillas") & (tab.k != "todo")]
        if not len(s):
            continue
        n = int(s.n_familias_mean.iloc[0])
        s = s.assign(kk=s.k.astype(int)).sort_values("kk")
        ax3.errorbar(s.kk, s.f1_macro_mean, yerr=s.f1_macro_std,
                     marker=mk, color=col, capsize=3,
                     label=f"{cur}: {n} familias, azar macro-F1 {1 / n:.3f}")
    ax3.set_xlabel("Tope de plantillas por familia en entrenamiento (k)")
    ax3.set_ylabel("macro-F1 (P2 con plantilla retenida)")
    ax3.set_title("(c) Tres conjuntos de familias\nNO son comparables entre si: "
                  "distinto numero de clases")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_b1_curva_notas.png", dpi=200)
    print(f"  Figura: {OUT / 'fig_b1_curva_notas.png'}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-estructural", type=int, default=None,
                    help="job SLURM de la tanda del detector (por defecto, el mayor)")
    ap.add_argument("--umbral-f1", type=float, default=0.98)
    ap.add_argument("--solo", choices=["2b", "2c", "b1"], default=None,
                    help="correr solo un bloque (por defecto, los tres)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (por defecto, 4_resultados/resumen_capitulo4). "
                         "Usar una carpeta NUEVA para no pisar el resumen canonico.")
    ap.add_argument("--curva-dir", type=Path, default=None,
                    help="carpeta con los CSV de B.1 (por defecto, "
                         "4_resultados/resultados_curva_notas). Apuntar a la salida NUEVA "
                         "de curva_aprendizaje_notas.py al re-medir sobre otra base.")
    args = ap.parse_args()
    if args.salida is not None:
        OUT = args.salida
    if args.solo in (None, "2b"):
        print("=" * 74)
        print("  EXPERIMENTO 2b — ablación del detector estructural")
        print("=" * 74)
        resumen_2b(args.job_estructural)
    if args.solo in (None, "2c"):
        print("\n" + "=" * 74)
        print("  EXPERIMENTO 2c — clasificador de bytes, por familia")
        print("=" * 74)
        resumen_2c(args.umbral_f1)
    if args.solo in (None, "b1"):
        print("\n" + "=" * 74)
        print("  B.1 — CURVA DE APRENDIZAJE DEL FRENTE DE NOTAS")
        print("=" * 74)
        resumen_b1(curva_dir=args.curva_dir)
    print(f"\nSalidas en: {OUT}")


if __name__ == "__main__":
    main()
