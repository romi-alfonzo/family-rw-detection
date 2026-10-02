#!/usr/bin/env python3
"""
auditoria_nombres_corpus.py — ¿El nombre de archivo de cada nota del corpus es GENUINO?

Prerequisito obligatorio de M.2/D.2. Sin esta auditoría el experimento del nombre de la nota
es circular y no se corre (`PLAN_MEJORAS.md` §D.2).

La distinción que importa NO es «el nombre contiene el nombre de la familia» — eso puede ser
señal legítima: `RyukReadMe.txt` lo bautizó así el propio malware, y es exactamente la señal
que usa ID Ransomware. La distinción es **QUIÉN puso ese nombre**:

  genuino             el nombre del corpus ES el que dejó el malware. Verificado por MD5
                      contra el repo de Lemmou (el archivo del repo se llama igual).
  genuino_renombrado  el CONTENIDO está verificado por MD5, pero el nombre del corpus fue
                      modificado por quien armó el corpus. El nombre genuino se recupera del
                      repo. Dos patrones detectados:
                        · sufijo `__N`     → artefacto de deduplicación (Info__3.hta)
                        · prefijo `lm_FAM_` → ⛔ CIRCULAR: mete el nombre de la familia en el
                                              nombre del archivo (lm_Cerber__HELP_DECRYPT_...)
  curador             el nombre lo inventó quien recolectó (blackbasta1.txt, pcrisk_cuba_1.txt,
                      note_pcrisk.txt, idr_*). No hay nombre genuino en el archivo mismo; si
                      existe, hay que traerlo de la fuente (advisory/MISP/pcrisk) — para eso
                      está `tabla_nombres_notas.py`.

Regla que se desprende y que M.2 tiene que respetar:
  **La vista «nombre de la nota» se construye SOLO con `nombre_para_m2`**, que es el nombre
  genuino cuando se conoce y vacío cuando no. Un nombre `curador` NUNCA entra como feature, y
  un `genuino_renombrado` entra con su nombre ORIGINAL, no con el del corpus.

Uso:
    python auditoria_nombres_corpus.py [--salida DIR]

Salidas: auditoria_nombres_corpus.csv · RESUMEN_auditoria_nombres.md
Solo lee: no modifica el corpus ni el manifiesto.
"""

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "3_datos" / "corpus_v2"
REPO_LEMMOU = RAIZ / "3_datos" / "fuentes_notas" / "RansomNoteFiles"
MANIFIESTO = RAIZ / "3_datos" / "manifiesto_corpus_v2.csv"
SALIDA_DEF = RAIZ / "3_datos" / "nombres_notas"

# Prefijos/sufijos que puso el curador del corpus, detectados al emparejar por MD5.
RE_SUFIJO_DEDUP = re.compile(r"__\d+(?=\.[^.]*$|$)")
RE_PREFIJO_LM = re.compile(r"^lm_([A-Za-z0-9]+)_", re.I)
# Nombres claramente inventados por quien recolectó (no son artefactos del malware).
RE_CURADOR = re.compile(
    r"^(note_|pcrisk_|idr_|hns_|lm_)|^[a-z]+\d+\.txt$|^[a-z]+_note\d*\.txt$", re.I)


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def indexar_repo(base: Path):
    """md5 -> conjunto de (carpeta, nombre_original) del repo de Lemmou."""
    idx = defaultdict(set)
    if not base.is_dir():
        return idx, 0
    n = 0
    for p in base.rglob("*"):
        if p.is_file():
            n += 1
            idx[md5(p)].add((p.parent.name, p.name))
    return idx, n


def clasificar(familia: str, archivo: str, originales):
    """Devuelve (procedencia, nombre_genuino, circular, detalle)."""
    if originales:
        nombres = sorted({n for _, n in originales})
        # Si alguno de los nombres del repo coincide exacto, el nombre del corpus es genuino.
        if archivo in nombres:
            return "genuino", archivo, False, "el repo trae el archivo con este mismo nombre"
        genuino = nombres[0]
        m = RE_PREFIJO_LM.match(archivo)
        if m:
            det = (f"⛔ el prefijo `lm_{m.group(1)}_` lo puso el curador y METE EL NOMBRE DE "
                   f"LA FAMILIA en el nombre del archivo; el nombre genuino es «{genuino}»")
            return "genuino_renombrado", genuino, True, det
        if RE_SUFIJO_DEDUP.search(archivo):
            base = RE_SUFIJO_DEDUP.sub("", archivo)
            det = (f"sufijo de deduplicación del curador; nombre genuino «{genuino}»"
                   + ("" if base == genuino else f" (base sin sufijo: «{base}»)"))
            return "genuino_renombrado", genuino, False, det
        return "genuino_renombrado", genuino, False, f"nombre del corpus distinto del original «{genuino}»"
    # Sin match en el repo: el nombre es del curador o de una transcripción.
    if RE_CURADOR.search(archivo):
        return "curador", "", True, "nombre inventado por quien recolectó; sin nombre genuino en el archivo"
    return "sin_verificar", "", False, "no hay fuente en disco que permita verificar el nombre"


def cargar_recuperados():
    """Nombres recuperados de la fuente de CADA nota (workflow del 2026-08-23).

    Son notas cuyo nombre el curador no preservo pero la fuente de esa nota SI declara
    (rotulo de pcrisk «Text presented in ... text file ("X")», paginas de id-ransomware).
    Cuentan como genuinos: el nombre sale de la fuente de ESA nota, no de una tabla de la
    familia — asignar el nombre documentado de la familia a cada una de sus notas seria
    convertir el nombre en una funcion de la etiqueta, o sea circularidad perfecta.
    """
    import json
    js = SALIDA_DEF / "nombres_por_nota_2026-08-23.json"
    rec = {}
    if js.is_file():
        with open(js, encoding="utf-8") as f:
            for r in json.load(f):
                nm = (r.get("nombre_archivo") or "").strip()
                if r.get("encontrado") and nm and nm != "SIN_ARCHIVO":
                    rec[(r["familia"], r["archivo_corpus"])] = (nm, r.get("url_usada", ""))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--salida", type=Path, default=SALIDA_DEF)
    args = ap.parse_args()

    repo, n_repo = indexar_repo(REPO_LEMMOU)
    recuperados = cargar_recuperados()
    print(f"Repo Lemmou: {n_repo} archivos, {len(repo)} hashes distintos")
    print(f"Nombres recuperados de la fuente de cada nota: {len(recuperados)}")

    manifiesto = {}
    with open(MANIFIESTO, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            manifiesto[(r["familia"], r["archivo"])] = r

    filas = []
    for fam_dir in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        for nota in sorted(p for p in fam_dir.iterdir() if p.is_file()):
            familia, archivo = fam_dir.name, nota.name
            m = md5(nota)
            originales = repo.get(m, set())
            proc, genuino, circular, detalle = clasificar(familia, archivo, originales)
            if proc in ("curador", "sin_verificar") and (familia, archivo) in recuperados:
                nm, url = recuperados[(familia, archivo)]
                proc, genuino, circular = "genuino_de_la_fuente", nm, False
                detalle = (f"el curador no preservo el nombre, pero la fuente de ESTA nota lo "
                           f"declara: «{nm}» ({url})")
            reg = manifiesto.get((familia, archivo), {})
            # ¿El nombre que se usaría en M.2 contiene el nombre de la familia?
            usable = genuino if proc in ("genuino", "genuino_renombrado",
                                         "genuino_de_la_fuente") else ""
            contiene_fam = bool(usable) and familia.lower()[:6] in re.sub(
                r"[^a-z0-9]", "", usable.lower())
            filas.append({
                "familia": familia,
                "archivo_corpus": archivo,
                "procedencia": proc,
                "nombre_genuino": genuino,
                "nombre_para_m2": usable,
                "circular": "si" if circular else "no",
                "nombre_genuino_contiene_familia": "si" if contiene_fam else "no",
                "variante_repo": " | ".join(sorted({c for c, _ in originales})),
                "md5_12": m[:12],
                "tipo_manifiesto": reg.get("tipo", ""),
                "fuente_manifiesto": (reg.get("fuente") or "").split("|")[0].strip(),
                "detalle": detalle,
            })

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_csv = args.salida / "auditoria_nombres_corpus.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(filas)

    cuenta = Counter(r["procedencia"] for r in filas)
    circulares = [r for r in filas if r["circular"] == "si"]
    usables = [r for r in filas if r["nombre_para_m2"]]
    por_fam = defaultdict(list)
    for r in filas:
        por_fam[r["familia"]].append(r)

    ruta_md = args.salida / "RESUMEN_auditoria_nombres.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Auditoría de procedencia de los nombres de archivo del corpus\n\n")
        f.write(f"**{len(filas)} notas auditadas.** Verificación por MD5 contra el repo de "
                f"Lemmou ({n_repo} archivos).\n\n")
        f.write("| Procedencia del nombre | Notas | ¿Sirve para M.2? |\n|---|---|---|\n")
        etiqueta = {
            "genuino": "✅ sí, tal cual",
            "genuino_renombrado": "✅ sí, con el nombre ORIGINAL del repo",
            "genuino_de_la_fuente": "✅ sí, nombre declarado por la fuente de esa nota",
            "curador": "⛔ no (sería circular)",
            "sin_verificar": "⚠️ no hasta traer el nombre de la fuente",
        }
        for k in ("genuino", "genuino_renombrado", "genuino_de_la_fuente", "curador", "sin_verificar"):
            if cuenta.get(k):
                f.write(f"| `{k}` | {cuenta[k]} | {etiqueta[k]} |\n")
        f.write(f"\n**Nombres usables en M.2: {len(usables)} de {len(filas)} notas "
                f"({100*len(usables)/len(filas):.0f} %).**\n\n")
        f.write("## ⛔ Circularidad detectada\n\n")
        if circulares:
            pref = [r for r in circulares if r["procedencia"] == "genuino_renombrado"]
            cur = [r for r in circulares if r["procedencia"] == "curador"]
            if pref:
                f.write(f"**{len(pref)} notas con el prefijo `lm_FAMILIA_` puesto por el "
                        f"curador**, que mete el nombre de la familia dentro del nombre del "
                        f"archivo. Si M.2 usara el nombre del corpus tal cual, estaría leyendo "
                        f"la etiqueta. El nombre genuino recuperado del repo NO la contiene:\n\n")
                for r in pref[:6]:
                    f.write(f"- `{r['archivo_corpus']}` → genuino `{r['nombre_genuino']}`\n")
                if len(pref) > 6:
                    f.write(f"- …y {len(pref)-6} más (ver CSV)\n")
            if cur:
                f.write(f"\n**{len(cur)} notas con nombre inventado por el recolector** "
                        f"(tipo `blackbasta1.txt`, `pcrisk_cuba_1.txt`): el nombre codifica la "
                        f"familia y no es un artefacto del malware. Quedan FUERA de la vista de "
                        f"M.2; su nombre genuino, si existe, hay que traerlo de una fuente "
                        f"externa (`tabla_nombres_notas.py`).\n")
        else:
            f.write("Ninguna.\n")
        f.write("\n## Nombres genuinos que SÍ contienen el nombre de la familia "
                "(señal legítima, no circular)\n\n")
        legit = [r for r in filas if r["nombre_genuino_contiene_familia"] == "si"]
        if legit:
            f.write("Los bautizó el propio malware, así que son señal real — es exactamente lo "
                    "que explota ID Ransomware. Se declara que la señal existe y de dónde "
                    "viene:\n\n")
            for r in sorted({(x["familia"], x["nombre_genuino"]) for x in legit}):
                f.write(f"- **{r[0]}**: `{r[1]}`\n")
        else:
            f.write("Ninguno entre los nombres verificados en disco.\n")
        f.write("\n## Cobertura por familia\n\n")
        f.write("| Familia | Notas | Con nombre usable | Procedencias |\n|---|---|---|---|\n")
        for fam in sorted(por_fam):
            rs = por_fam[fam]
            us = sum(1 for r in rs if r["nombre_para_m2"])
            procs = ", ".join(f"{k}:{v}" for k, v in
                             Counter(r["procedencia"] for r in rs).most_common())
            f.write(f"| {fam} | {len(rs)} | {us} | {procs} |\n")

    print(f"CSV:     {ruta_csv}  ({len(filas)} notas)")
    print(f"Resumen: {ruta_md}")
    print(f"Procedencias: {dict(cuenta)}")
    print(f"Usables en M.2: {len(usables)}/{len(filas)} · circulares detectadas: {len(circulares)}")


if __name__ == "__main__":
    main()
