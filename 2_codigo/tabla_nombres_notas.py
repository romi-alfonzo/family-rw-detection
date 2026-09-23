#!/usr/bin/env python3
"""
tabla_nombres_notas.py — Tabla citable de NOMBRES DE ARCHIVO de nota por familia.

Insumo de M.2 (nombre+extensión de la nota como vista) y de la tabla de la tesis. Consolida
dos orígenes distintos y NO los mezcla, porque su valor probatorio es distinto:

  a) `misp`      — el mapeo auditado del catálogo MISP (`mapeo_misp_familias.py`). Solo se
                   leen las filas cuyo dictamen sea INCLUIR; el dictamen manual de Romina
                   (`dictamen_romina`) MANDA sobre el borrador si está completado.
  b) `busqueda`  — recolección con fuente citable por nombre (advisory oficial, CERT,
                   id-ransomware, pcrisk, vendor), en JSON, con URL y evidencia por fila.

Reglas que el script hace cumplir, no sugiere:
  - Cada nombre lleva SU fuente. Un nombre sin `url` no entra en la tabla: sale a la lista
    de rechazados con motivo (misma disciplina que el corpus de notas).
  - `SIN_ARCHIVO` es un RESULTADO, no un hueco: la familia que muestra su mensaje en una
    ventana (CryptoLocker 2013, Jigsaw) se registra así, con su fuente.
  - Se marca la `procedencia` de cada nombre para M.2: `genuino` (documentado por una fuente
    externa) frente a `curador` (lo puso quien armó un repositorio, p. ej. `conti1.txt`).
    **La etiqueta `curador` NUNCA entra como feature** — es el bloqueo original de D.2.
  - Se detectan y reportan las COLISIONES: un mismo nombre de nota usado por más de una
    familia (p. ej. YOUR_FILES_ARE_ENCRYPTED.*), que son el límite declarado de M.2.

Uso:
    python tabla_nombres_notas.py --busqueda <hallazgos.json> [--salida DIR]

El JSON de búsqueda espera {"filas": [{familia, nombre_archivo, tipo_fuente,
organismo_o_autor, titulo_documento, id_documento, fecha, url, evidencia, confianza,
observaciones}, ...]}; acepta también una lista de esos objetos en la raíz.

Salidas: tabla_nombres_notas.csv · rechazados_nombres.csv · RESUMEN_nombres.md
Solo lee sus insumos: no toca el corpus, el manifiesto ni los resultados.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MAPEO_DEF = RAIZ / "3_datos" / "misp_ransomware_galaxy" / "mapeo_borrador" / "mapeo_misp_borrador.csv"
SALIDA_DEF = RAIZ / "3_datos" / "nombres_notas"

FAMILIAS = [
    "AVOSLOCKER", "BADRABBIT", "BLACKBASTA", "BLACKCAT", "BLACKMATTER", "CERBER",
    "CHIMERA", "CLOP", "CONTI", "CRYPTOLOCKER", "CUBA", "DARKSIDE", "DHARMA",
    "GANDCRAB", "HELLOKITTY", "JIGSAW", "LOCKBIT", "LORENZ", "MAZE", "MEDUZALOCKER",
    "NETWALKER", "NOTPETYA", "PHOBOS", "RANSOMEXX", "RYUK", "SODINOKIBI", "SUNCRYPT",
    "TESLACRYPT", "WANNACRY", "WASTEDLOCKER",
]

# Peso probatorio, para ordenar la tabla y para el resumen.
ORDEN_FUENTE = {"advisory_oficial": 0, "cert": 1, "id-ransomware": 2, "pcrisk": 3,
                "vendor": 4, "misp": 5}


def clave_nombre(nombre: str) -> str:
    """Normaliza para detectar colisiones, de forma DELIBERADAMENTE conservadora.

    Solo se unifica lo que no cambia el nombre del archivo: mayúsculas, espacios sobrantes
    y la forma concreta del placeholder (`<victim>`, `{ID}`, `%random%` → `*`). NO se borran
    guiones bajos ni signos, porque `READ_ME_!!!.TXT` y `README.txt` **son archivos
    distintos**: borrarlos producía colisiones falsas (Clop y RansomEXX apareciendo como
    `readme.txt`), y una colisión falsa en la tesis es una afirmación incorrecta. Se prefiere
    perder una colisión real a inventar una.
    """
    s = nombre.strip().lower()
    s = re.sub(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]|%[a-z_]+%", "*", s)  # placeholders → comodín
    s = re.sub(r"\s+", "", s)
    return s


def leer_mapeo(ruta: Path):
    """Nombres de nota de las entradas MISP dictaminadas INCLUIR."""
    filas, avisos = [], []
    if not ruta.is_file():
        return filas, [f"mapeo MISP no encontrado en {ruta}: la tabla sale solo con la búsqueda"]
    with open(ruta, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            dictamen = (r.get("dictamen_romina") or "").strip().upper() or \
                       (r.get("dictamen_borrador") or "").strip().upper()
            auditado = bool((r.get("dictamen_romina") or "").strip())
            if dictamen != "INCLUIR":
                continue
            nombres = [n.strip() for n in (r.get("ransomnotes_filenames") or "").split("|") if n.strip()]
            for n in nombres:
                filas.append({
                    "familia": r["familia"], "nombre_archivo": n, "origen": "misp",
                    "tipo_fuente": "misp", "organismo_o_autor": "MISP Project",
                    "titulo_documento": f"Ransomware galaxy — entrada «{r['entrada_misp']}»",
                    "id_documento": r.get("uuid", ""), "fecha": r.get("fecha_misp", ""),
                    "url": r.get("primera_ref", ""), "evidencia": "meta.ransomnotes-filenames",
                    "confianza": "media", "procedencia": "genuino",
                    "auditado": "si" if auditado else "no",
                    "observaciones": "nombre del catálogo MISP; conviene respaldarlo con advisory",
                })
    if filas and not any(f["auditado"] == "si" for f in filas):
        avisos.append("El mapeo MISP todavía NO está auditado (`dictamen_romina` vacío): "
                      "las filas de origen `misp` son BORRADOR y no son citables aún.")
    return filas, avisos


def leer_busqueda(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    crudas = datos if isinstance(datos, list) else datos.get("filas", [])
    filas = []
    for r in crudas:
        filas.append({
            "familia": (r.get("familia") or "").strip().upper(),
            "nombre_archivo": (r.get("nombre_archivo") or "").strip(),
            "origen": "busqueda",
            "tipo_fuente": (r.get("tipo_fuente") or "").strip(),
            "organismo_o_autor": r.get("organismo_o_autor", ""),
            "titulo_documento": r.get("titulo_documento", ""),
            "id_documento": r.get("id_documento", ""),
            "fecha": r.get("fecha", ""),
            "url": (r.get("url") or "").strip(),
            "evidencia": r.get("evidencia", ""),
            "confianza": r.get("confianza", ""),
            "procedencia": "genuino",
            "auditado": "verificado" if r.get("verificado") else "no",
            "observaciones": r.get("observaciones", ""),
        })
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--busqueda", type=Path, help="JSON con los hallazgos de la recolección")
    ap.add_argument("--mapeo", type=Path, default=MAPEO_DEF)
    ap.add_argument("--salida", type=Path, default=SALIDA_DEF)
    args = ap.parse_args()

    filas, avisos = leer_mapeo(args.mapeo)
    if args.busqueda:
        filas += leer_busqueda(args.busqueda)
    if not filas:
        raise SystemExit("Sin insumos: pasá --busqueda y/o generá el mapeo MISP primero.")

    # --- Filtro: sin fuente no entra (SIN_ARCHIVO sí necesita fuente, es una afirmación).
    aceptadas, rechazadas = [], []
    for f in filas:
        if f["familia"] not in FAMILIAS:
            f["motivo_rechazo"] = f"familia «{f['familia']}» fuera de las 30 canónicas"
            rechazadas.append(f)
        elif not f["nombre_archivo"]:
            f["motivo_rechazo"] = "sin nombre de archivo"
            rechazadas.append(f)
        elif not f["url"]:
            f["motivo_rechazo"] = "sin URL de fuente (regla del proyecto: toda cifra/dato citable)"
            rechazadas.append(f)
        else:
            aceptadas.append(f)

    # --- Deduplicar por (familia, nombre normalizado, url): la misma fuente no cuenta 2 veces.
    vistas, tabla = set(), []
    for f in sorted(aceptadas, key=lambda x: (x["familia"], ORDEN_FUENTE.get(x["tipo_fuente"], 9))):
        k = (f["familia"], clave_nombre(f["nombre_archivo"]), f["url"])
        if k in vistas:
            continue
        vistas.add(k)
        tabla.append(f)

    # --- Colisiones: mismo nombre en más de una familia (límite declarado de M.2).
    por_nombre = defaultdict(set)
    for f in tabla:
        if f["nombre_archivo"].upper() != "SIN_ARCHIVO":
            por_nombre[clave_nombre(f["nombre_archivo"])].add(f["familia"])
    colisiones = {n: sorted(fs) for n, fs in por_nombre.items() if len(fs) > 1}

    # --- Cuántas fuentes independientes respaldan cada (familia, nombre)
    fuentes_por_par = defaultdict(set)
    for f in tabla:
        fuentes_por_par[(f["familia"], clave_nombre(f["nombre_archivo"]))].add(f["url"])

    args.salida.mkdir(parents=True, exist_ok=True)
    cols = ["familia", "nombre_archivo", "origen", "tipo_fuente", "organismo_o_autor",
            "titulo_documento", "id_documento", "fecha", "url", "evidencia", "confianza",
            "procedencia", "auditado", "n_fuentes_del_par", "colision_con", "observaciones"]
    ruta_csv = args.salida / "tabla_nombres_notas.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=";", extrasaction="ignore")
        w.writeheader()
        for r in tabla:
            k = clave_nombre(r["nombre_archivo"])
            r["n_fuentes_del_par"] = len(fuentes_por_par[(r["familia"], k)])
            otras = [x for x in colisiones.get(k, []) if x != r["familia"]]
            r["colision_con"] = " | ".join(otras)
            w.writerow(r)

    ruta_rech = args.salida / "rechazados_nombres.csv"
    with open(ruta_rech, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["familia", "nombre_archivo", "origen",
                                          "motivo_rechazo", "url", "observaciones"],
                           delimiter=";", extrasaction="ignore")
        w.writeheader()
        for r in rechazadas:
            w.writerow(r)

    # --- Resumen
    por_fam = defaultdict(list)
    for r in tabla:
        por_fam[r["familia"]].append(r)
    ruta_md = args.salida / "RESUMEN_nombres.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Nombres de archivo de nota por familia — tabla citable\n\n")
        for a in avisos:
            f.write(f"> ⚠️ {a}\n\n")
        con_dato = [x for x in FAMILIAS if por_fam.get(x)]
        con_oficial = [x for x in FAMILIAS
                       if any(r["tipo_fuente"] in ("advisory_oficial", "cert") for r in por_fam.get(x, []))]
        f.write(f"**Cobertura: {len(con_dato)}/30 familias con al menos un nombre citable** "
                f"({len(con_oficial)}/30 con advisory oficial o CERT). "
                f"Filas: {len(tabla)} aceptadas · {len(rechazadas)} rechazadas.\n\n")
        f.write("| Familia | Nombres | Mejor fuente | ¿2+ fuentes? |\n|---|---|---|---|\n")
        for fam in FAMILIAS:
            rs = por_fam.get(fam, [])
            if not rs:
                f.write(f"| {fam} | — | ⛔ sin dato | — |\n")
                continue
            nombres = sorted({r["nombre_archivo"] for r in rs})
            mejor = min(rs, key=lambda r: ORDEN_FUENTE.get(r["tipo_fuente"], 9))
            dobles = sum(1 for k, u in fuentes_por_par.items() if k[0] == fam and len(u) >= 2)
            muestra = "; ".join(nombres[:4]) + ("…" if len(nombres) > 4 else "")
            f.write(f"| {fam} | {muestra} | {mejor['tipo_fuente']} "
                    f"({mejor['organismo_o_autor']}) | {dobles} de "
                    f"{len(nombres)} |\n")
        f.write("\n## Colisiones de nombre entre familias (límite declarado de M.2)\n\n")
        if colisiones:
            for n, fams in sorted(colisiones.items()):
                ejemplo = next(r["nombre_archivo"] for r in tabla if clave_nombre(r["nombre_archivo"]) == n)
                f.write(f"- `{ejemplo}` → {', '.join(fams)}\n")
            f.write("\nEl nombre de nota **no es unívoco**: reportar estas colisiones al "
                    "declarar el resultado de M.2.\n")
        else:
            f.write("Ninguna detectada en esta tabla.\n")
        sin_archivo = [r for r in tabla if r["nombre_archivo"].upper() == "SIN_ARCHIVO"]
        if sin_archivo:
            f.write("\n## Familias sin archivo de nota (resultado, no hueco)\n\n")
            for r in sin_archivo:
                f.write(f"- **{r['familia']}**: {r['evidencia'] or 'muestra su mensaje en ventana'} "
                        f"— {r['organismo_o_autor']}, {r['url']}\n")

    print(f"Tabla:      {ruta_csv}  ({len(tabla)} filas)")
    print(f"Rechazados: {ruta_rech}  ({len(rechazadas)} filas)")
    print(f"Resumen:    {ruta_md}")
    for a in avisos:
        print(f"[!] {a}")


if __name__ == "__main__":
    main()
