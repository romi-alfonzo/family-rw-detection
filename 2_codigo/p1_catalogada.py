#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p1_catalogada.py -- réplica controlada: la cascada cuando la plantilla ESTÁ en el catálogo.

POR QUÉ. P1 reparte notas, no plantillas, y 76 de las 149 notas pertenecen a plantillas de una
sola nota: para ellas la plantilla nunca está en entrenamiento. Así, la etiqueta «plantilla ya
catalogada» de P1 es exacta solo para el 41 % de las decisiones (diagnostico_p1.py, 50 semillas).
Este protocolo mide lo que esa etiqueta promete, garantizándolo por construcción.

PROTOCOLO P1cat. Por semilla s (rng = 30_000 + s):
  - cada plantilla con 2 o más notas se parte en dos mitades no vacías, A y B, al azar;
  - pliegue 1: prueba = todas las A, entrenamiento = todas las B + todas las notas de plantillas
    de una sola nota; pliegue 2: al revés;
  - así cada nota evaluada tiene al menos una instancia de su plantilla en entrenamiento, y las
    notas únicas entran siempre al catálogo (como en una herramienta en producción), pero nunca
    se evalúan;
  - las 30 familias están siempre disponibles como clases, aunque solo se evalúen las que tienen
    alguna plantilla repetida.
Misma representación, mismo clasificador, misma capa de reglas que abstencion_notas.py.

PUERTA DE ENTRADA. En la misma corrida y con el mismo código se evalúa también P1 tal como está
publicado (StratifiedKFold, 2 pliegues, random_state = s). Tiene que reproducir la cascada en
0,8866 de exactitud y 0,8592 de macro-F1, y el texto solo en 0,8389 y 0,7889
(_log_m3_149_P1.txt). Si no, se aborta y no se reporta nada.

PREREGISTRO (escrito y commiteado ANTES de correr, 2026-10-01). Se reporta lo que dé.
  PC-1  La cascada bajo P1cat acierta 0,97 o más. Fundamento: bajo P1, las notas con alguna
        hermana en entrenamiento se aciertan 0,9773, y P1cat además deja todas las notas únicas
        en el catálogo.
  PC-2  El texto solo bajo P1cat acierta 0,90 o más, porque siempre tiene en entrenamiento una
        variante casi idéntica de la nota evaluada.
  PC-3  La capa de reglas resuelve el 80 % o más de las notas evaluadas (bajo P1, 0,877 de las
        notas con hermana en entrenamiento se resolvieron por regla).
ADVERTENCIA AL CITAR. La exactitud es comparable con la de los otros protocolos. El macro-F1
NO: se promedia solo sobre las familias evaluadas (las que tienen alguna plantilla repetida).

Uso:  python p1_catalogada.py [--n-semillas 50]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PUERTA = {"cascada": (0.8866, 0.8592), "texto": (0.8389, 0.7889)}
TOL = 0.0005
OUT = ab.RAIZ / "4_resultados" / "resultados_p1_catalogada"


def particiones_p1cat(grupos, rng):
    tam = Counter(grupos)
    unicas = [i for i in range(len(grupos)) if tam[grupos[i]] == 1]
    miembros = defaultdict(list)
    for i, g in enumerate(grupos):
        if tam[g] >= 2:
            miembros[g].append(i)
    A, B = [], []
    for g in sorted(miembros):
        idx = rng.permutation(miembros[g])
        corte = int(rng.integers(1, len(idx)))
        A.extend(idx[:corte])
        B.extend(idx[corte:])
    A, B, U = np.array(A), np.array(B), np.array(unicas)
    return [(np.concatenate([B, U]), A), (np.concatenate([A, U]), B)]


def evaluar(splits, textos_arr, y, iocs, nombres_nota, s):
    """Devuelve, para las notas evaluadas: indices, prediccion de texto, prediccion de cascada,
    y si decidio la regla."""
    idx, p_txt, p_cas, reg = [], [], [], []
    for tr, te in splits:
        vec = ab.vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = ab.obtener_modelos(s)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        top1 = clf.classes_[np.argmax(clf.decision_function(Xte), axis=1)]
        d = ab.dicc_privados(tr, iocs, nombres_nota, y)
        for k, i in enumerate(te):
            claves = set(iocs[i])
            if nombres_nota[i]:
                claves.add(("[NOMBRE]", nombres_nota[i]))
            fams = set()
            for c in claves:
                if c in d:
                    fams |= d[c]
            aplica = len(fams) == 1
            idx.append(i)
            p_txt.append(top1[k])
            p_cas.append(next(iter(fams)) if aplica else top1[k])
            reg.append(aplica)
    return np.array(idx), np.array(p_txt), np.array(p_cas), np.array(reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-semillas", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
    grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
    textos_arr = np.array(textos, dtype=object)
    y = np.asarray(y)
    grupos = np.asarray(grupos)
    iocs = [set(ab.extraer_marcadores(t)) for t in textos]
    nom_aud = ab.cargar_nombres()
    nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]
    tam = Counter(grupos)
    evaluables = sorted(i for i in range(len(y)) if tam[grupos[i]] >= 2)
    fams_eval = sorted(set(y[evaluables]))
    print(f"Notas: {len(y)} | plantillas: {len(tam)} | evaluables en P1cat: {len(evaluables)} "
          f"notas de {sum(1 for v in tam.values() if v >= 2)} plantillas y {len(fams_eval)} familias")

    m = {k: defaultdict(list) for k in ("P1", "P1cat")}
    por_familia = defaultdict(lambda: [0, 0])
    for s in range(args.n_semillas):
        cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
        corridas = {
            "P1": list(cv.split(textos_arr, y)),
            "P1cat": particiones_p1cat(grupos, np.random.default_rng(30_000 + s)),
        }
        for prot, splits in corridas.items():
            idx, p_txt, p_cas, reg = evaluar(splits, textos_arr, y, iocs, nombres_nota, s)
            yt = y[idx]
            for nombre, pred in (("texto", p_txt), ("cascada", p_cas)):
                m[prot][nombre + "_ac"].append(accuracy_score(yt, pred))
                m[prot][nombre + "_f1"].append(f1_score(yt, pred, average="macro", zero_division=0))
            m[prot]["cobertura_regla"].append(reg.mean())
            m[prot]["acierto_regla"].append(accuracy_score(yt[reg], p_cas[reg]) if reg.any() else np.nan)
            if prot == "P1cat":
                for fam, ok in zip(yt, p_cas == yt):
                    por_familia[fam][0] += int(ok)
                    por_familia[fam][1] += 1
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{args.n_semillas} semillas", flush=True)

    r = {p: {k: float(np.nanmean(v)) for k, v in d.items()} for p, d in m.items()}

    print("\n" + "-" * 78 + "\n  PUERTA DE ENTRADA: P1 publicado, con este mismo código\n" + "-" * 78)
    ok = True
    for sist, (ac0, f10) in PUERTA.items():
        ac, f1 = r["P1"][sist + "_ac"], r["P1"][sist + "_f1"]
        bien = abs(ac - ac0) <= TOL and abs(f1 - f10) <= TOL
        ok &= bien
        print(f"  {sist:<8} exactitud {ac:.4f} (publicada {ac0}) | macro-F1 {f1:.4f} "
              f"(publicado {f10})  {'OK' if bien else 'NO REPRODUCE'}")
    if not ok:
        sys.exit("ABORTADO: el código no reproduce P1 publicado. No se reporta nada.")

    rc = r["P1cat"]
    print("\n" + "=" * 78 + f"\n  P1cat -- plantilla en el catálogo garantizada ({args.n_semillas} semillas)\n"
          + "=" * 78)
    print(f"  texto solo : exactitud {rc['texto_ac']:.4f} | macro-F1 {rc['texto_f1']:.4f} "
          f"(sobre {len(fams_eval)} familias evaluadas)")
    print(f"  cascada    : exactitud {rc['cascada_ac']:.4f} | macro-F1 {rc['cascada_f1']:.4f}")
    print(f"  capa de reglas: resuelve {rc['cobertura_regla']:.4f} de las notas, acierta "
          f"{rc['acierto_regla']:.4f}")

    print("\n  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    for cod, desc, val, umbral in (
            ("PC-1", "cascada >= 0,97", rc["cascada_ac"], 0.97),
            ("PC-2", "texto solo >= 0,90", rc["texto_ac"], 0.90),
            ("PC-3", "cobertura de reglas >= 0,80", rc["cobertura_regla"], 0.80)):
        print(f"  [{'CUMPLE' if val >= umbral else 'FALLA '}] {cod} {desc:<30} {val:.4f}")

    print("\n  Familias con menor acierto de la cascada en P1cat:")
    for fam, (ac, n) in sorted(por_familia.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]:
        print(f"    {fam:<14} {ac/n:.4f}  ({n} decisiones)")

    with open(OUT / "p1_catalogada_resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protocolo", "sistema", "exactitud", "macro_f1", "cobertura_regla", "acierto_regla"])
        for p in ("P1", "P1cat"):
            for sist in ("texto", "cascada"):
                w.writerow([p, sist, round(r[p][sist + "_ac"], 4), round(r[p][sist + "_f1"], 4),
                            round(r[p]["cobertura_regla"], 4) if sist == "cascada" else "",
                            round(r[p]["acierto_regla"], 4) if sist == "cascada" else ""])
    with open(OUT / "p1_catalogada_por_familia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["familia", "aciertos", "decisiones", "exactitud"])
        for fam, (ac, n) in sorted(por_familia.items()):
            w.writerow([fam, ac, n, round(ac / n, 4)])
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
