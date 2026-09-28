#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capa_extension_misp.py -- la extension de cifrado resuelta con el CATALOGO DEL TUTOR.

DE DONDE SALE. La capa de extension medida el 2026-09-28 (`capa_extension_cifrado.py`) dio
Delta = +0,0000 exacto, y la causa NO fue el metodo sino una restriccion del protocolo: la regla
solo dispara si el MISMO valor ya se vio en otra plantilla del ENTRENAMIENTO. De las 12 notas que
conservan su extension, solo 6 cumplen eso. **Techo duro 6/149 = 0,0403**, y techo oraculo
+0,0038 aun con conocimiento perfecto.

LO QUE CAMBIA ACA. Se sustituye el diccionario aprendido del corpus por el **catalogo MISP
«Ransomware»**, la fuente que envio el Prof. Cappo el 2026-08-20
(`3_datos/misp_ransomware_galaxy/misp_galaxy_ransomware_2026-08-20.json`). Trae **735 extensiones
distintas, 673 de ellas asociadas a UNA sola familia**. Con eso la extension mencionada en una
nota se resuelve **sin necesidad de haberla visto en entrenamiento**, que era la unica barrera.

LO QUE ESTO CAMBIA EN LA NATURALEZA DEL SISTEMA, y hay que declararlo. Deja de ser un sistema
puramente aprendido del corpus: incorpora un **catalogo de referencia externo**. No es una
trampa -- es exactamente lo que hace ID Ransomware, la herramienta de produccion con la que este
trabajo se compara -- pero cambia lo que se puede afirmar, y por eso se reporta como una capa
aparte y se mide su aporte por separado.

RIESGO DE CIRCULARIDAD, declarado: el catalogo MISP es comunitario y pudo alimentarse de las
mismas fuentes publicas de las que salieron algunas notas del corpus. No hay forma de descartarlo
desde aca. Lo que si se controla es que **el catalogo no ve las etiquetas del corpus**: se usa
tal como vino, sin ajustarlo.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

M1. PUERTA DE ENTRADA. Sin la capa, el sistema reproduce macro-F1 0,7417 y exactitud 0,8123
    (tolerancia 0,01). Si no, ABORTA.
M2. La cobertura de la capa MISP supera el techo duro de la version aprendida (0,0403). Es la
    razon de ser del experimento: si no lo supera, el catalogo no aporta alcance.
M3. El acierto donde la capa aplica es >= 0,90. El catalogo asocia extension a familia con
    autoridad externa; si acertara poco, el mapeo no sirve o las extensiones extraidas del texto
    no son las que el catalogo registra.
M4. LA PREDICCION PRINCIPAL: Delta macro-F1 > 0 con IC 95 % que excluye el cero. Puede fallar:
    el techo oraculo medido sobre las 12 notas con extension era +0,0038, asi que el margen es
    chico por construccion. Si el Delta es nulo, la conclusion es que **la barrera no era el
    diccionario sino cuantas notas conservan la extension**, y eso cierra la via del todo.
M5. Las familias que ganan son las que mencionan su extension y hoy no la aprovechan:
    SODINOKIBI, LORENZ, NETWALKER. GANDCRAB ya esta en recall 1,0000 y no puede ganar.
M6. Sobre el corpus EXTENDIDO de 106 familias la cobertura es MAYOR que sobre el nucleo: hay
    mas familias con extension registrada en el catalogo.
=============================================================================================

Uso:  python capa_extension_misp.py [--n-semillas 50] [--extendido] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import accuracy_score, f1_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from capa_extension_cifrado import extraer_extensiones
from clasificador_notas_v2 import obtener_modelos, vectorizador
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
MISP = RAIZ / "3_datos" / "misp_ransomware_galaxy" / "misp_galaxy_ransomware_2026-08-20.json"
OUT_DEF = RAIZ / "4_resultados" / "resultados_capa_extension_misp"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01
TECHO_APRENDIDO = 0.0403


def normalizar(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def cargar_misp():
    """extension -> familia normalizada, solo las que el catalogo asocia a UNA familia."""
    d = json.load(open(MISP, encoding="utf-8"))
    ext2fams = {}
    for v in d["values"]:
        m = v.get("meta", {})
        nombres = [v["value"]] + list(m.get("synonyms", []))
        nn = [normalizar(x) for x in nombres if normalizar(x)]
        if not nn:
            continue
        can = nn[0]
        for e in m.get("extensions", []):
            e = e.strip().lower().lstrip(".")
            if e:
                ext2fams.setdefault(e, set()).add(can)
    privadas = {e: next(iter(f)) for e, f in ext2fams.items() if len(f) == 1}
    return privadas, ext2fams


def ic(v):
    v = np.asarray(v, float)
    m, nn = float(np.mean(v)), len(v)
    s = float(np.std(v, ddof=1)) if nn > 1 else 0.0
    h = t_dist.ppf(0.975, nn - 1) * (s / np.sqrt(nn)) if nn > 1 else 0.0
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=OUT_DEF)
    ap.add_argument("--n-semillas", type=int, default=50)
    ap.add_argument("--sin-puerta", action="store_true")
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  LA EXTENSION DE CIFRADO, RESUELTA CON EL CATALOGO MISP DEL TUTOR")
    print("=" * 78)
    priv, todas = cargar_misp()
    print(f"Catalogo MISP: {len(todas)} extensiones distintas | "
          f"{len(priv)} apuntan a UNA sola familia")

    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    fam_norm = {normalizar(f): f for f in familias}
    print(f"Notas: {n} | Familias: {len(familias)} | Semillas: {args.n_semillas}\n")

    # ---------- que resuelve el catalogo sobre nuestras notas ----------
    resuelve = np.empty(n, dtype=object)
    detalle = []
    for i, t in enumerate(textos):
        for e in extraer_extensiones(t):
            e = e.strip().lower().lstrip(".")
            f_misp = priv.get(e)
            if f_misp and f_misp in fam_norm:          # solo si la familia esta en el catalogo
                resuelve[i] = fam_norm[f_misp]
                detalle.append(dict(archivo=archivos[i], familia_real=y[i],
                                    extension=e, familia_misp=fam_norm[f_misp],
                                    acierta=fam_norm[f_misp] == y[i]))
                break
    dd = pd.DataFrame(detalle)
    if len(dd):
        dd.to_csv(OUT / "misp_resoluciones.csv", index=False, encoding="utf-8-sig")
    cob = float((resuelve != None).sum()) / n                      # noqa: E711
    ac = float(dd.acierta.mean()) if len(dd) else np.nan
    print(f"=== LO QUE EL CATALOGO RESUELVE POR SI SOLO ===")
    print(f"  notas con extension que el MISP mapea a una familia del catalogo: "
          f"{int((resuelve != None).sum())} de {n}  (cobertura {cob:.4f})")       # noqa: E711
    print(f"  techo duro de la version APRENDIDA del corpus: {TECHO_APRENDIDO}")
    print(f"  acierto de esas resoluciones: {ac:.4f}" if len(dd) else "  (ninguna)")
    if len(dd):
        print(dd.head(15).to_string(index=False))

    # ---------- evaluacion ----------
    print("\nEvaluando ...")
    p_base = np.empty((args.n_semillas, n), dtype=object)
    p_misp = np.empty((args.n_semillas, n), dtype=object)
    for s in range(args.n_semillas):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(vec.transform(textos_arr[te]))
            d = dicc_privados(tr, iocs, nombres_nota, y)
            for k, i in enumerate(te):
                r = regla(i, d, iocs, nombres_nota)
                base = pt[k] if r is None else r
                p_base[s, i] = base
                # la capa MISP va DESPUES de las reglas aprendidas y ANTES del texto
                if r is not None:
                    p_misp[s, i] = r
                elif resuelve[i] is not None:
                    p_misp[s, i] = resuelve[i]
                else:
                    p_misp[s, i] = pt[k]
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas}")

    f1_b = np.array([f1_score(y, p_base[s], average="macro", labels=familias, zero_division=0)
                     for s in range(args.n_semillas)])
    f1_m = np.array([f1_score(y, p_misp[s], average="macro", labels=familias, zero_division=0)
                     for s in range(args.n_semillas)])
    ac_b = np.array([accuracy_score(y, p_base[s]) for s in range(args.n_semillas)])
    ac_m = np.array([accuracy_score(y, p_misp[s]) for s in range(args.n_semillas)])

    print("\n" + "-" * 78)
    print("  M1 -- PUERTA DE ENTRADA")
    print("-" * 78)
    print(f"  base: macro-F1 {f1_b.mean():.4f} vs {CANON_F1} | exactitud {ac_b.mean():.4f} vs {CANON_ACC}")
    ok = abs(f1_b.mean() - CANON_F1) <= TOL and abs(ac_b.mean() - CANON_ACC) <= TOL
    if not ok and not args.sin_puerta:
        sys.exit("ABORTADO (M1).")
    print("  OK\n" if ok else "  FUERA DE TOLERANCIA\n")

    d_f1 = f1_m - f1_b
    d_ac = ac_m - ac_b
    mf, lof, hif = ic(d_f1)
    ma, loa, hia = ic(d_ac)
    res = pd.DataFrame([
        dict(sistema="cascada actual", macro_f1=round(float(f1_b.mean()), 4),
             exactitud=round(float(ac_b.mean()), 4)),
        dict(sistema="cascada + capa MISP", macro_f1=round(float(f1_m.mean()), 4),
             exactitud=round(float(ac_m.mean()), 4))])
    res.to_csv(OUT / "misp_resumen.csv", index=False, encoding="utf-8-sig")
    print("=== RESULTADO ===")
    print(res.to_string(index=False))
    print(f"\n  Delta macro-F1 : {mf:+.4f} [{lof:+.4f}; {hif:+.4f}]  "
          f"{int((d_f1 > 0).sum())}/{args.n_semillas} semillas")
    print(f"  Delta exactitud: {ma:+.4f} [{loa:+.4f}; {hia:+.4f}]  "
          f"{int((d_ac > 0).sum())}/{args.n_semillas} semillas")

    # por familia
    ff = []
    for f in familias:
        idx = np.where(y == f)[0]
        ff.append(dict(familia=f, n_notas=len(idx),
                       base=round(float(np.mean([(p_base[s, idx] == f).mean()
                                                 for s in range(args.n_semillas)])), 4),
                       con_misp=round(float(np.mean([(p_misp[s, idx] == f).mean()
                                                     for s in range(args.n_semillas)])), 4)))
    dff = pd.DataFrame(ff)
    dff["delta"] = (dff.con_misp - dff.base).round(4)
    dff.sort_values("delta", ascending=False).to_csv(OUT / "misp_por_familia.csv",
                                                     index=False, encoding="utf-8-sig")
    mueven = dff[dff.delta != 0].sort_values("delta", ascending=False)
    print("\n=== FAMILIAS QUE SE MUEVEN ===")
    print(mueven.to_string(index=False) if len(mueven) else "  (ninguna)")

    print("\n" + "=" * 78)
    print("  VEREDICTO DEL PREREGISTRO")
    print("=" * 78)
    chk = [
        ("M1 puerta de entrada", ok, f"{f1_b.mean():.4f} / {ac_b.mean():.4f}"),
        ("M2 cobertura supera el techo aprendido (0,0403)", cob > TECHO_APRENDIDO,
         f"{cob:.4f}"),
        ("M3 acierto donde aplica >= 0,90", (ac >= 0.90) if len(dd) else False,
         f"{ac:.4f}" if len(dd) else "sin resoluciones"),
        ("M4 Delta macro-F1 > 0 con IC que excluye el cero", lof > 0,
         f"{mf:+.4f} [{lof:+.4f}; {hif:+.4f}]"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<48} {det}")
    if lof <= 0:
        print("\n  LECTURA (fijada en M4): el Delta no excluye el cero. La barrera NO era el")
        print("  diccionario sino CUANTAS NOTAS CONSERVAN LA EXTENSION: las fuentes publican")
        print("  las notas saneadas. Cierra la via, ahora con catalogo externo y todo.")
    print(f"\n  AL CITAR: esta capa usa un CATALOGO EXTERNO (MISP, enviado por el tutor el")
    print("  2026-08-20). El sistema deja de ser puramente aprendido del corpus.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
