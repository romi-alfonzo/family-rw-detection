"""¿Qué notas de las fuentes nuevas se pueden SUMAR a las 30 familias canónicas?

Extrae los textos de nota que traen las fuentes nuevas, se queda con los que pertenecen a
una de las 30 familias del corpus, y para cada uno aplica **el criterio del proyecto**
(`agrupar_neardups`, coseno char_wb 3-5 > 0,90): solo cuenta el que aporta una PLANTILLA
nueva. Una nota que colapsa con una que ya está no suma nada (B.1).

PRIORIDAD, que sale de la curva re-medida sobre 149: el corte está en **3 plantillas** por
familia y pasado ahí el macro-F1 BAJA de forma medible. Así que interesan las familias que
hoy tienen menos de 3, y NO interesa sumar a las que ya llegaron.

⚠️ LA TRAMPA DEL HOMÓNIMO, que en este proyecto ya pegó tres veces (Medusa vs MedusaLocker,
Crypt0l0cker vs CryptoLocker, HelloKitty vs Dharma): el emparejamiento por nombre se hace
con una tabla EXPLÍCITA y las dudosas se reportan aparte, sin incorporarlas. Nada entra al
corpus desde este script: solo dictamina y deja el candidato en disco para revisión humana.

Uso:
    python candidatas_de_fuentes_nuevas.py --misp <json> [--zsig <md>] [--volcar <carpeta>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, TFIDF_CHAR, UMBRAL_NEARDUP,
                                   agrupar_neardups, cargar_corpus)
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Mapeo EXPLÍCITO de nombre externo -> familia canónica. Solo lo que es la misma familia.
# Se escribe a mano a propósito: un normalizador automático es como se cuela un homónimo.
ALIAS = {
    "avoslocker": "AVOSLOCKER", "bad rabbit": "BADRABBIT", "badrabbit": "BADRABBIT",
    "black basta": "BLACKBASTA", "blackbasta": "BLACKBASTA",
    "blackcat": "BLACKCAT", "alphv": "BLACKCAT", "blackcat (alphv)": "BLACKCAT",
    "blackmatter": "BLACKMATTER", "cerber": "CERBER", "chimera": "CHIMERA",
    "clop": "CLOP", "cl0p": "CLOP", "conti": "CONTI",
    "cryptolocker": "CRYPTOLOCKER", "cuba": "CUBA", "darkside": "DARKSIDE",
    "dharma": "DHARMA", "crysis": "DHARMA", "gandcrab": "GANDCRAB",
    "hellokitty": "HELLOKITTY", "hello kitty": "HELLOKITTY",
    "jigsaw": "JIGSAW", "lockbit": "LOCKBIT", "lorenz": "LORENZ", "maze": "MAZE",
    "medusalocker": "MEDUZALOCKER", "netwalker": "NETWALKER", "mailto": "NETWALKER",
    "notpetya": "NOTPETYA", "petya/notpetya": "NOTPETYA",
    "phobos": "PHOBOS", "ransomexx": "RANSOMEXX", "ryuk": "RYUK",
    "sodinokibi": "SODINOKIBI", "revil": "SODINOKIBI", "revil (sodinokibi)": "SODINOKIBI",
    "sodinokibi (revil)": "SODINOKIBI",
    "suncrypt": "SUNCRYPT", "teslacrypt": "TESLACRYPT",
    "wannacry": "WANNACRY", "wanacry": "WANNACRY", "wannacrypt": "WANNACRY",
    "wastedlocker": "WASTEDLOCKER",
}
# Nombres que PARECEN de una familia canónica y NO lo son. Precedentes reales del proyecto.
TRAMPAS = {
    "medusa": "Medusa != MedusaLocker (error ya cometido: 2 notas retiradas el 2026-08-19)",
    "crypt0l0cker": "Crypt0l0cker = TorrentLocker, NO CryptoLocker (nota retirada)",
    "teslarvng": "TeslaRVNG no es TeslaCrypt",
    "phobos impersonating": "campaña, no familia",
    "conti locker": "revisar: puede ser variante o imitador",
}


def texto_util(t, minimo=60):
    t = str(t)
    return not t.startswith("http") and len(t.strip()) >= minimo


def de_misp(ruta: Path):
    d = json.loads(ruta.read_text(encoding="utf-8"))
    CLAVES = ("ransomnotes", "ransomenotes")
    for v in d.get("values", []):
        meta = v.get("meta") or {}
        nombre = str(v.get("value", "")).strip()
        for k in CLAVES:
            for t in (meta.get(k) or []) if isinstance(meta.get(k), list) else \
                     ([meta[k]] if k in meta else []):
                if texto_util(t):
                    yield nombre, str(t), "MISP"


def de_zsig(ruta: Path):
    md = ruta.read_text(encoding="utf-8", errors="replace")
    for p in re.split(r"\n##\s+", md)[1:]:
        cab, _, cuerpo = p.partition("\n")
        if texto_util(cuerpo):
            yield cab.strip(), cuerpo, "Zsigovits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misp", type=Path, default=None)
    ap.add_argument("--zsig", type=Path, default=None)
    ap.add_argument("--volcar", type=Path, default=None,
                    help="carpeta donde dejar las candidatas NUEVAS para revisión humana")
    args = ap.parse_args()

    textos, y, archivos, _ = cargar_corpus(CORPUS_DIR)
    grupos = agrupar_neardups(textos, UMBRAL_NEARDUP)[0]
    fams = sorted(set(y))
    plant = {f: len({g for g, ff in zip(grupos, y) if ff == f}) for f in fams}
    print("=" * 84)
    print("  CANDIDATAS DE LAS FUENTES NUEVAS PARA LAS 30 FAMILIAS CANÓNICAS")
    print("=" * 84)
    print("corpus: %d notas · %d familias · %d plantillas" % (len(textos), len(fams),
                                                             len(set(grupos))))
    prioritarias = sorted(f for f in fams if plant[f] < 3)
    print("familias por debajo del corte de 3 plantillas (%d): %s"
          % (len(prioritarias), ", ".join(prioritarias)))

    cands, dudosas = [], []
    for fuente_it, ruta in ((de_misp, args.misp), (de_zsig, args.zsig)):
        if ruta is None:
            continue
        if not ruta.is_file():
            sys.exit("ABORTA: no existe %s" % ruta)
        for nombre, texto, fuente in fuente_it(ruta):
            clave = nombre.lower().strip()
            if clave in TRAMPAS:
                dudosas.append((nombre, fuente, TRAMPAS[clave]))
                continue
            fam = ALIAS.get(clave)
            if fam is None:
                # ¿el nombre contiene una familia canónica? se reporta, no se acepta
                for k, v in ALIAS.items():
                    if k in clave and len(k) > 5:
                        dudosas.append((nombre, fuente,
                                        "parece %s pero el nombre no es exacto" % v))
                        break
                continue
            cands.append((fam, nombre, texto, fuente))

    print("\ncandidatas mapeadas a una de las 30: %d" % len(cands))
    print("nombres dudosos NO aceptados: %d" % len(dudosas))

    if not cands:
        print("\nNo hay candidatas. Nada que evaluar.")
        return

    # ¿aporta plantilla nueva? mismo criterio que verificar_nota_nueva.py
    base = list(textos)
    vec = TfidfVectorizer(**TFIDF_CHAR)
    X = vec.fit_transform(base + [c[2] for c in cands])
    sim = (X @ X.T).toarray()
    n = len(base)

    print("\n" + "-" * 84)
    print("  %-14s %-26s %-10s %8s %-22s %s"
          % ("familia", "nombre en la fuente", "fuente", "coseno", "vecino más cercano", "¿aporta?"))
    print("-" * 84)
    nuevas_por_fam = {}
    for i, (fam, nombre, texto, fuente) in enumerate(cands):
        fila = sim[n + i, :n]
        j = int(np.argmax(fila))
        cos = float(fila[j])
        # también contra las otras candidatas ya aceptadas de la misma familia
        aporta = cos <= UMBRAL_NEARDUP
        for k, (f2, _n2, t2, _s2) in enumerate(cands[:i]):
            if f2 == fam and sim[n + i, n + k] > UMBRAL_NEARDUP:
                aporta = False
                break
        marca = "SÍ" if aporta else "no (colapsa)"
        if aporta:
            nuevas_por_fam.setdefault(fam, []).append((nombre, fuente, texto))
        pri = " ★" if fam in prioritarias else ""
        print("  %-14s %-26s %-10s %8.4f %-22s %s%s"
              % (fam, nombre[:26], fuente, cos, y[j] + "/" + Path(archivos[j]).name[:12],
                 marca, pri))

    print("-" * 84)
    print("\n== RESUMEN: plantillas nuevas que se podrían sumar")
    if not nuevas_por_fam:
        print("   ninguna: todas colapsan con lo que ya hay.")
    for fam in sorted(nuevas_por_fam):
        pri = "★ PRIORITARIA (hoy %d plantillas)" % plant[fam] if fam in prioritarias \
              else "(hoy %d plantillas: pasado el corte, NO conviene)" % plant[fam]
        print("   %-14s +%d  %s" % (fam, len(nuevas_por_fam[fam]), pri))

    if dudosas:
        print("\n== NOMBRES DUDOSOS, NO aceptados (revisión humana)")
        for nombre, fuente, motivo in dudosas[:20]:
            print("   %-28s %-10s %s" % (nombre[:28], fuente, motivo))

    if args.volcar and nuevas_por_fam:
        args.volcar.mkdir(parents=True, exist_ok=True)
        n_esc = 0
        for fam, lista in nuevas_por_fam.items():
            d = args.volcar / fam
            d.mkdir(exist_ok=True)
            for k, (nombre, fuente, texto) in enumerate(lista, 1):
                seguro = re.sub(r"[^A-Za-z0-9_.-]", "_", nombre)[:40]
                (d / ("%s_%s_%d.txt" % (fuente.lower(), seguro, k))).write_text(
                    texto, encoding="utf-8")
                n_esc += 1
        print("\n%d candidatas escritas en %s" % (n_esc, args.volcar))
        print("NO están en el corpus: hay que revisarlas a mano y registrar procedencia.")


if __name__ == "__main__":
    main()
