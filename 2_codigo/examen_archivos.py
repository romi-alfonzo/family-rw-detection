# -*- coding: utf-8 -*-
"""Examen externo del frente de archivos: el modelo del Exp. 2g frente a archivos cifrados de otra procedencia.

Pedido de Romina (2026-11-01): probar en local el modelo entrenado en NapierOne con archivos cifrados reales que
NO salgan de NapierOne (otra ejecución u otra versión del programa), unos diez por familia. Mide la robustez de
cada capa: el modelo de contenido (bytes + estructura) y el completo (con la forma de la extensión).

Carpeta del examen: <raiz>/<FAMILIA>/<archivos cifrados>, con los nombres de archivo tal como quedaron tras el
cifrado (la extensión agregada importa). Cada archivo debe estar en manifiesto.csv (columnas: familia, archivo,
fuente, url, fecha) con su procedencia; si falta, se avisa y se evalúa igual.

Uso: python examen_archivos.py <raiz del examen> <modelos_exp2g.joblib> [--salida carpeta]
Salida: examen_por_archivo.csv, examen_por_familia.csv y examen_resumen.txt.
Prueba local: python examen_archivos.py <carpeta sintética> <joblib de --prueba> --salida <carpeta>
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
from exp2e_estructura_bytes import N_HEAD, N_TAIL, leer, rasgos_estructura  # noqa: E402
from exp2g_nombre_robusto import forma_de_la_extension  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def cargar_examen(raiz):
    Xb, Xe, y, nombres, rutas = [], [], [], [], []
    for d in sorted(p for p in Path(raiz).iterdir() if p.is_dir()):
        fam = d.name.upper()
        for p in sorted(q for q in d.iterdir() if q.is_file() and q.name.lower() != "readme.md"):
            head, cola, cabp, colap, medios, tam = leer(p)
            Xb.append(head + cola)
            Xe.append(rasgos_estructura(cabp, colap, medios, tam))
            y.append(fam)
            nombres.append(p.name)
            rutas.append(str(p.relative_to(raiz)))
    if not y:
        sys.exit(f"no hay archivos en {raiz}: se esperan subcarpetas por familia")
    Xb = np.frombuffer(b"".join(Xb), dtype=np.uint8).reshape(len(Xb), N_HEAD + N_TAIL).astype(np.float32)
    Xe = np.nan_to_num(np.asarray(Xe, dtype=np.float32))
    return Xb, Xe, forma_de_la_extension(nombres), np.array(y), nombres, rutas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz", type=Path)
    ap.add_argument("modelos", type=Path)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    out = args.salida or (_AQUI.parent / "4_resultados" / "resultados_examen_archivos")
    out.mkdir(parents=True, exist_ok=True)

    paq = joblib.load(args.modelos)
    assert paq["n_head"] == N_HEAD and paq["n_tail"] == N_TAIL, "el modelo se entrenó con otra ventana de bytes"
    Xb, Xe, Xr, y, nombres, rutas = cargar_examen(args.raiz)
    assert Xe.shape[1] == paq["n_estructura"], "distinto número de rasgos estructurales"

    manifiesto = {}
    m = args.raiz / "manifiesto.csv"
    if m.exists():
        with open(m, encoding="utf-8", newline="") as f:
            for fila in csv.DictReader(f):
                manifiesto[(fila["familia"].upper(), fila["archivo"])] = fila.get("fuente", "")
    sin_proc = [(f, n) for f, n in zip(y, nombres) if (f, n) not in manifiesto]

    familias_modelo = set(paq["familias"])
    fuera = sorted(set(y) - familias_modelo)
    filas = []
    for nombre, X in (("contenido", np.hstack([Xb, Xe])), ("completo", np.hstack([Xb, Xe, Xr]))):
        clf = paq["modelos"][nombre]
        proba = clf.predict_proba(X)
        pred = clf.classes_[proba.argmax(1)]
        for i in range(len(y)):
            filas.append(dict(modelo=nombre, familia=y[i], archivo=rutas[i], prediccion=pred[i],
                              confianza=round(float(proba[i].max()), 4), acierto=int(pred[i] == y[i]),
                              fuente=manifiesto.get((y[i], nombres[i]), "")))
    df = pd.DataFrame(filas)
    df.to_csv(out / "examen_por_archivo.csv", index=False, encoding="utf-8")
    porfam = (df.groupby(["modelo", "familia"]).agg(archivos=("acierto", "size"), aciertos=("acierto", "sum"))
              .reset_index())
    porfam["acierto"] = (porfam.aciertos / porfam.archivos).round(4)
    porfam.to_csv(out / "examen_por_familia.csv", index=False, encoding="utf-8")

    lineas = [f"EXAMEN EXTERNO DEL FRENTE DE ARCHIVOS  ·  {len(y)} archivos de {len(set(y))} familias  ·  "
              f"modelo entrenado con {paq['n_entrenamiento']} archivos ({paq['por_familia']}/familia)"]
    if fuera:
        lineas.append(f"  familias del examen que el modelo no conoce (siempre fallan): {', '.join(fuera)}")
    if sin_proc:
        lineas.append(f"  ⚠ {len(sin_proc)} archivos sin procedencia en manifiesto.csv")
    for nombre in ("contenido", "completo"):
        d = df[df.modelo == nombre]
        dm = d[d.familia.isin(familias_modelo)]
        lineas.append(f"\n  modelo {nombre}: acierto {dm.acierto.mean():.4f} sobre {len(dm)} archivos de familias "
                      f"conocidas (donde el azar es {1 / len(familias_modelo):.3f}); confianza media {dm.confianza.mean():.3f}")
        errores = defaultdict(int)
        for _, r in dm[dm.acierto == 0].iterrows():
            errores[(r.familia, r.prediccion)] += 1
        for (f, p), n in sorted(errores.items(), key=lambda kv: -kv[1])[:12]:
            lineas.append(f"    {f} → {p}: {n}")
    lineas.append("\n  por familia (acierto contenido / completo):")
    tabla = porfam.pivot(index="familia", columns="modelo", values="acierto")
    for fam, r in tabla.iterrows():
        n = int(porfam[(porfam.familia == fam) & (porfam.modelo == "completo")].archivos.iloc[0])
        lineas.append(f"    {fam:14s} {r.get('contenido', float('nan')):.2f} / {r.get('completo', float('nan')):.2f}   (n={n})")
    texto = "\n".join(lineas)
    print(texto)
    (out / "examen_resumen.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
