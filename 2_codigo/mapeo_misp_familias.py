#!/usr/bin/env python3
"""
mapeo_misp_familias.py — Borrador auditable del mapeo familia→entrada del catálogo MISP.

Prerequisito de M.2 (nombre+extensión de la nota como vista) y de la tabla citable de
extensiones: saber QUÉ entradas del catálogo MISP corresponden a cada una de las 30
familias canónicas, y cuáles son homónimos, imitadores o falsos positivos del nombre.

Método:
  1. Se buscan candidatas por alias (igualdad exacta de `value`/`synonyms` normalizados,
     y además por subcadena). Los alias son A PROPÓSITO amplios (p. ej. `medusa`, `tesla`,
     `wcry`): la meta es ARRASTRAR los homónimos conocidos para dictaminarlos a la vista,
     no esconderlos. La subcadena produce falsos positivos reales (`wcry` matchea
     Pe-WCRY-pt y Shado-WCRY-ptor; `wncry` matchea Unkno-WNCRY-pted; `clop` matchea
     cy-CLOP-s): por eso el mapeo NO puede ser automático.
  2. Cada candidata lleva un DICTAMEN BORRADOR (INCLUIR / EXCLUIR / REVISAR) con motivo,
     escrito revisando la descripción de la entrada y los hallazgos ya documentados en
     ESTADO_TESIS.md (Medusa≠MedusaLocker AA25-071A, Crypt0l0cker=TorrentLocker,
     homónimos de CryptoLocker, imitadores de WannaCry/Cerber/GandCrab).
  3. La columna `dictamen_romina` queda VACÍA: la auditoría final es humana. El borrador
     no es citable; el mapeo auditado sí.

Salidas (en --salida, por defecto 3_datos/misp_ransomware_galaxy/mapeo_borrador/):
  - mapeo_misp_borrador.csv  (UTF-8 con BOM, para abrir directo en Excel)
  - RESUMEN_mapeo.md         (cobertura por familia tras aplicar el borrador)

Solo lee el catálogo: NO modifica el corpus, el manifiesto ni ningún resultado.
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CATALOGO_DEF = RAIZ / "3_datos" / "misp_ransomware_galaxy" / "misp_galaxy_ransomware_2026-08-20.json"
SALIDA_DEF = RAIZ / "3_datos" / "misp_ransomware_galaxy" / "mapeo_borrador"

# Alias de búsqueda por familia. Amplios a propósito (ver docstring): los homónimos
# tienen que ENTRAR al mapeo para poder dictaminarlos con motivo, no quedar invisibles.
ALIAS = {
    "AVOSLOCKER": ["avoslocker"],
    "BADRABBIT": ["badrabbit"],
    "BLACKBASTA": ["blackbasta"],
    "BLACKCAT": ["blackcat", "alphv"],
    "BLACKMATTER": ["blackmatter"],
    "CERBER": ["cerber"],
    "CHIMERA": ["chimera"],
    "CLOP": ["clop", "cl0p"],
    "CONTI": ["conti"],
    "CRYPTOLOCKER": ["cryptolocker"],
    "CUBA": ["cuba"],
    "DARKSIDE": ["darkside"],
    "DHARMA": ["dharma", "crysis"],
    "GANDCRAB": ["gandcrab"],
    "HELLOKITTY": ["hellokitty"],
    "JIGSAW": ["jigsaw"],
    "LOCKBIT": ["lockbit"],
    "LORENZ": ["lorenz"],
    "MAZE": ["maze"],
    "MEDUZALOCKER": ["medusalocker", "medusa"],
    "NETWALKER": ["netwalker", "mailto"],
    "NOTPETYA": ["notpetya", "petya"],
    "PHOBOS": ["phobos"],
    "RANSOMEXX": ["ransomexx", "defray"],
    "RYUK": ["ryuk"],
    "SODINOKIBI": ["sodinokibi", "revil"],
    "SUNCRYPT": ["suncrypt"],
    "TESLACRYPT": ["teslacrypt", "tesla"],
    "WANNACRY": ["wannacry", "wcry", "wanacry", "wncry"],
    "WASTEDLOCKER": ["wastedlocker", "wasted"],
}

# Entradas que el barrido por alias NO captura pero que DEBEN estar en el mapeo como
# controles documentados (verificación adversarial 2026-08-22). El caso clave es
# TorrentLocker: `Crypt0L0cker` se escribe con ceros, así que no contiene la subcadena
# `cryptolocker` — exactamente la trampa que la tesis ya resolvió en el corpus.
EXTRAS = [
    ("CRYPTOLOCKER", "TorrentLocker"),
    ("CRYPTOLOCKER", "Crypt0L0cker"),
    ("AVOSLOCKER", "Avos"),
    ("NOTPETYA", "GoldenEye Ransomware"),
    ("NOTPETYA", "PetrWrap Ransomware"),
    ("LORENZ", "SZ40"),
]

# Dictámenes borrador, clave (FAMILIA, value exacto de la entrada MISP).
# INCLUIR = la entrada ES la familia canónica; EXCLUIR = homónimo/imitador/falso positivo;
# REVISAR = hace falta decisión humana (alcance o atribución dudosa).
# Verificados adversarialmente contra el JSON completo el 2026-08-22 (subagente):
# 74/76 resistieron; se corrigieron mailto (INCLUIR→REVISAR) y cerberimposter
# (REVISAR→EXCLUIR), 4 motivos, y se agregaron las 6 entradas de EXTRAS.
DICTAMENES = {
    ("AVOSLOCKER", "AvosLocker"): ("INCLUIR", "entrada canónica (advisory FBI/FinCEN); fn GET_YOUR_FILES_BACK.TXT"),
    ("AVOSLOCKER", "Avos"): ("EXCLUIR", "entrada duplicada: el catálogo declara 'Avos' como synonym de AvosLocker y además existe como entrada propia vacía (solo un .onion); sin metadatos útiles"),
    ("BADRABBIT", "Bad Rabbit"): ("INCLUIR", "entrada canónica (Cisco Talos); sin fn/ext en MISP"),
    ("BLACKBASTA", "BlackBasta"): ("INCLUIR", "entrada canónica; fn readme.txt, ext .basta"),
    ("BLACKCAT", "BlackCat"): ("INCLUIR", "entrada canónica; syn ALPHV/Noberus; sin fn/ext en MISP"),
    ("BLACKMATTER", "Darkside"): ("EXCLUIR", "MISP fusiona BlackMatter como SINÓNIMO de Darkside; la tesis las trata como 2 familias — BLACKMATTER queda SIN entrada propia; usar la misma entrada para 2 clases induciría colisión artificial en M.2"),
    ("CERBER", "CerberTear Ransomware"): ("EXCLUIR", "imitador basado en HiddenTear que usa branding de Cerber; su ext .cerber es imitada"),
    ("CERBER", "Cerber"): ("INCLUIR", "entrada canónica; 12 fn, ext .cerber/.cerber2/.cerber3"),
    ("CERBER", "Cerberos"): ("EXCLUIR", "descripción genérica ('Ransomware') y sin metadatos; nombre distinto; sin evidencia de ser la familia"),
    ("CERBER", "Fake Cerber"): ("EXCLUIR", "imitador declarado en el propio nombre"),
    ("CERBER", "cerbersyslock"): ("EXCLUIR", "la propia descripción lo llama 'cryptoransomware imposter' con branding estilo Cerber"),
    ("CERBER", "cerberimposter"): ("EXCLUIR", "la propia entrada lo resuelve: 'It does not reuse the original Cerber codebase; instead it borrows branding' — imitador de marca; fn __$$RECOVERY_README$$__.html y ext .locked, distintos de todos los canónicos"),
    ("CHIMERA", "Chimera"): ("INCLUIR", "fn YOUR_FILES_ARE_ENCRYPTED.HTML/TXT coincide con la nota pcrisk del corpus; OJO: syn 'Quimera Crypter'/'Pashka' son de atribución dudosa (Pashka es otra familia de 2020)"),
    ("CLOP", "Clop"): ("INCLUIR", "entrada canónica; 4 fn (ClopReadMe.txt...), 8 ext"),
    ("CLOP", "clop torrents"): ("EXCLUIR", "artefacto de ransomlook: el sitio de leaks/torrents del propio grupo Clop como entrada aparte; duplicado sin metadatos útiles, no una familia distinta"),
    ("CLOP", "cyclops"): ("EXCLUIR", "FALSO POSITIVO de subcadena (cy-CLOP-s): Cyclops/Knight es familia sin relación"),
    ("CONTI", "Conti"): ("INCLUIR", "entrada canónica; ext .conti; MISP no trae nombre de nota para Conti"),
    ("CRYPTOLOCKER", "CryptoLocker"): ("INCLUIR", "entrada del original 2013; ⚠️ sus ext .encrypted/.ENC CONTRADICEN la fuente primaria SecureWorks (el original reemplaza el archivo; ver pregunta abierta en ESTADO_TESIS) y son casi idénticas a las de la entrada TorrentLocker (.Encrypted/.enc): sospecha de contaminación entre entradas del catálogo — NO citar esas extensiones sin verificación"),
    ("CRYPTOLOCKER", "TorrentLocker"): ("EXCLUIR", "CONTROL de la decisión de la tesis: Crypt0L0cker = TorrentLocker ≠ CryptoLocker 2013 (syn Crypt0L0cker/CryptoFortress/Teerac); sus fn documentados incluyen HOW_TO_RESTORE_FILES.html — el nombre EXACTO de la nota retirada del corpus el 2026-08-19 (confirmación independiente de esa limpieza) — y DECRYPT_INSTRUCTIONS.html + 8 versiones multiidioma; el barrido por subcadena no la captura porque Crypt0L0cker se escribe con ceros"),
    ("CRYPTOLOCKER", "Crypt0L0cker"): ("EXCLUIR", "entrada vacía duplicada de TorrentLocker (sin metadatos); mismo control que la anterior"),
    ("CRYPTOLOCKER", "CryptoLocker by NTK Ransomware"): ("EXCLUIR", "homónimo de 2017 sin relación con el original"),
    ("CRYPTOLOCKER", "DynA-Crypt Ransomware"): ("EXCLUIR", "homónimo (syn 'DynA CryptoLocker'); familia distinta de 2017"),
    ("CRYPTOLOCKER", "CryptoLocker3 Ransomware"): ("EXCLUIR", "homónimo auto-declarado: syn 'Fake CryptoLocker'"),
    ("CRYPTOLOCKER", "MSN CryptoLocker Ransomware"): ("EXCLUIR", "homónimo de 2016 sin relación con el original"),
    ("CRYPTOLOCKER", "PClock3 Ransomware"): ("EXCLUIR", "homónimo auto-declarado: syn 'CryptoLocker clone'"),
    ("CRYPTOLOCKER", "CryptoLocker 1.0.0"): ("EXCLUIR", "homónimo sin metadatos; el original 2013 no versionaba así"),
    ("CRYPTOLOCKER", "CryptoLocker 5.1"): ("EXCLUIR", "homónimo sin metadatos; ídem"),
    ("CRYPTOLOCKER", "FakeCryptoLocker"): ("EXCLUIR", "imitador declarado en el propio nombre"),
    ("CRYPTOLOCKER", "Acroware Cryptolocker Ransomware"): ("EXCLUIR", "screenlocker que NO cifra (lo dice su descripción); solo usa el nombre"),
    ("CRYPTOLOCKER", "CryptolockerEmulator"): ("EXCLUIR", "emulador/homónimo sin metadatos"),
    ("CRYPTOLOCKER", "CryptoLockerEU 2016"): ("EXCLUIR", "homónimo de 2016 sin relación con el original"),
    ("CRYPTOLOCKER", "GoCryptoLocker"): ("EXCLUIR", "homónimo sin metadatos"),
    ("CRYPTOLOCKER", "MNS CryptoLocker"): ("EXCLUIR", "homónimo sin metadatos"),
    ("CUBA", "Cuba"): ("INCLUIR", "entrada canónica; syn COLDDRAW/Fidel; sin fn/ext en MISP"),
    ("DARKSIDE", "Darkside"): ("INCLUIR", "entrada canónica; ⚠️ MISP le pone BlackMatter como sinónimo — la tesis las separa; no reusar esta entrada para BLACKMATTER"),
    ("DHARMA", "Dharma Ransomware"): ("INCLUIR", "la entrada rica del linaje: fn Info.hta / FILES ENCRYPTED.txt coinciden con el corpus; 21 ext, 4 textos"),
    ("DHARMA", "Virus-Encoder"): ("INCLUIR", "es CrySiS (syn), el linaje temprano de Dharma; fn 'How to decrypt your data.txt', ext .xtbl/.DHARMA; declarar que se cuenta dentro de DHARMA"),
    ("DHARMA", "Phobos"): ("EXCLUIR", "matcheó DHARMA solo por su syn 'Java NotDharma'; es la familia PHOBOS (clase propia de la tesis)"),
    ("DHARMA", "Crysis XTBL"): ("EXCLUIR", "duplicado nominal de CrySiS sin ningún metadato; el linaje ya está cubierto por Virus-Encoder"),
    ("DHARMA", "Java NotDharma"): ("EXCLUIR", "el propio nombre dice NotDharma"),
    ("DHARMA", "Hunt"): ("REVISAR", "la desc lo llama 'variant of the Dharma/CrySIS family' (RaaS de afiliados): fn info-hunt.txt, ext .hunt; decidir si variantes de afiliado cuentan como DHARMA"),
    ("DHARMA", "dharma"): ("INCLUIR", "entrada duplicada de la familia (buena descripción, sin metadatos); tratar como la misma clase"),
    ("GANDCRAB", "GandCrab"): ("INCLUIR", "entrada canónica; fn GDCB-DECRYPT.txt / CRAB-Decrypt.txt, 4 textos de nota"),
    ("GANDCRAB", "Jokeroo"): ("EXCLUIR", "RaaS distinto que se promocionó como GandCrab: syn 'Fake GandCrab'"),
    ("HELLOKITTY", "HelloKitty"): ("INCLUIR", "entrada canónica; ⚠️ syn FiveHands (vendors lo tratan como sucesor/relacionado); sin fn/ext en MISP"),
    ("JIGSAW", "Jigsaw"): ("INCLUIR", "19 ext que incluyen .AFD y .fun (las variantes alemana y francesa del corpus); syn CryptoHitMan (reskin documentado); sin fn en MISP"),
    ("LOCKBIT", "LockBit"): ("INCLUIR", "entrada canónica; fn Restore-My-Files.txt, ext .abcd/.LockBit"),
    ("LOCKBIT", "Lockbit3"): ("EXCLUIR", "entrada de versión sin metadatos útiles (solo refs); la familia ya está cubierta"),
    ("LOCKBIT", "lockbit4"): ("EXCLUIR", "ídem"),
    ("LOCKBIT", "lockbit5"): ("EXCLUIR", "ídem"),
    ("LORENZ", "Lorenz Ransomware"): ("INCLUIR", "entrada de la familia (doble extorsión desde 2021); sin fn/ext en MISP"),
    ("LORENZ", "lorenz"): ("INCLUIR", "entrada duplicada (desc Tesorion/NoMoreRansom); misma clase"),
    ("LORENZ", "SZ40"): ("REVISAR", "entrada sin descripción, metadatos ni refs: el catálogo no alcanza para dictaminar; duda externa a verificar con fuente: el malware del grupo Lorenz se conoce como 'Lorenz.sZ40'"),
    ("MAZE", "Maze"): ("INCLUIR", "entrada canónica; la desc menciona nota en .txt y .htm pero MISP no estructura fn/ext"),
    ("MEDUZALOCKER", "MedusaLocker"): ("INCLUIR", "la entrada más rica del catálogo: 11 fn (incluye HOW_TO_RECOVER_DATA.html del corpus), 36 ext"),
    ("MEDUZALOCKER", "Ako"): ("REVISAR", "Ako/MedusaReborn: algunos vendors lo vinculan a MedusaLocker y otros lo tratan aparte; trae fn ako-readme.txt y 6 textos; NO incorporar sin fuente que diga MedusaLocker (misma regla que Medusa)"),
    ("MEDUZALOCKER", "medusa"): ("EXCLUIR", "TRAMPA DOCUMENTADA: Medusa ≠ MedusaLocker (FBI/CISA AA25-071A); su fn !!!READ_ME_MEDUSA!!!.txt es el de las 2 notas RETIRADAS del corpus el 2026-08-19 — control negativo del mapeo"),
    ("NETWALKER", "Netwalker"): ("INCLUIR", "entrada canónica; sin metadatos en MISP"),
    ("NETWALKER", "mailto"): ("REVISAR", "el catálogo NO dice en ningún lado que mailto = NetWalker (la entrada solo trae el ref ransomlook.io/group/mailto, sin desc ni synonyms); Mailto fue el nombre inicial de NetWalker según vendors, pero hace falta fuente citable — mismo rasero que Ako"),
    ("NOTPETYA", "Mischa"): ("EXCLUIR", "Petya+Mischa 2016: familia anterior, no NotPetya 2017"),
    ("NOTPETYA", "Petya"): ("EXCLUIR", "es el Petya ORIGINAL 2016 (syn GoldenEye); NotPetya 2017 NO tiene entrada propia en el catálogo (verificado exhaustivamente: cero resultados para ExPetr/Nyetya/Petna/EternalPetya/PetrWrap/DiskCoder en value+synonyms de las 2135 entradas); su fn YOUR_FILES_ARE_ENCRYPTED.TXT es de Petya — la nota de NotPetya es README.TXT (fuente: CCN-CERT, ya registrada)"),
    ("NOTPETYA", "GoldenEye Ransomware"): ("EXCLUIR", "linaje Petya 2016 (dic-2016), existe como entrada propia además del synonym de Petya; no es NotPetya"),
    ("NOTPETYA", "PetrWrap Ransomware"): ("EXCLUIR", "PetrWrap original de marzo-2017 (refs securelist mar-2017), ANTERIOR a NotPetya; algunos vendors llamaron PetrWrap a NotPetya en jun-2017, pero esta entrada MISP no es eso"),
    ("PHOBOS", "Phobos"): ("INCLUIR", "entrada canónica; ext .phobos en la desc; sin fn estructurado"),
    ("PHOBOS", "PhobosImposter"): ("EXCLUIR", "imitador declarado en el propio nombre"),
    ("RANSOMEXX", "RansomEXX"): ("INCLUIR", "entrada canónica; syn Defray777; fn TXDOT_READ_ME!.Txt y <empresa>_READ_ME!.txt (nota nombrada por víctima), 6 ext"),
    ("RANSOMEXX", "Defray (Glushkov)"): ("EXCLUIR", "Defray 2017 original, linaje anterior; sin metadatos; RansomEXX ya cubierto por su entrada"),
    ("RYUK", "Ryuk ransomware"): ("INCLUIR", "entrada canónica (el value trae 'ransomware' pegado); fn RyukReadMe.txt; la desc conecta con Hermes (→ Exp. A.5a)"),
    ("SODINOKIBI", "Sodinokibi"): ("INCLUIR", "entrada canónica; syn REvil; sin fn/ext en MISP"),
    ("SUNCRYPT", "SunCrypt"): ("INCLUIR", "⚠️ colisión real de nombre de nota a declarar en M.2: YOUR_FILES_ARE_ENCRYPTED.* aparece en CUATRO entradas — Chimera (.HTML y .TXT), SunCrypt (.HTML), Mischa (.HTML) y Petya (.TXT) — el nombre de nota no es unívoco entre familias"),
    ("TESLACRYPT", "TeslaCrypt 0.x - 2.2.0"): ("INCLUIR", "entrada versionada de la familia; syn AlphaCrypt; 2 fn, 8 ext"),
    ("TESLACRYPT", "TeslaCrypt 3.0+"): ("INCLUIR", "entrada versionada; 4 ext (.micro/.xxx/.ttt/.mp3); '4.0+ no agrega extensión'"),
    ("TESLACRYPT", "TeslaCrypt 4.1A"): ("INCLUIR", "entrada versionada; 17 fn (patrones RECOVER/RESTORE con aleatorios)"),
    ("TESLACRYPT", "TeslaCrypt 4.2"): ("INCLUIR", "entrada versionada; 17 fn (casi idénticos a 4.1A: deduplicar al consolidar)"),
    ("TESLACRYPT", "TeslaWare"): ("EXCLUIR", "FALSO POSITIVO del alias 'tesla': familia sin relación"),
    ("WANNACRY", "Wcry Ransomware"): ("REVISAR", "Wcry feb-2017: varios vendors lo consideran el predecesor directo / v1 de WannaCry; ext .wcry; decidir si cuenta como WANNACRY"),
    ("WANNACRY", "WannaCry"): ("INCLUIR", "entrada canónica; syn WanaCrypt0r/WCRY; MISP NO trae fn — los nombres @Please_Read_Me@.txt / !Please Read Me!.txt salen de otras fuentes"),
    ("WANNACRY", "AutoWannaCryV2"): ("EXCLUIR", "imitador sin metadatos"),
    ("WANNACRY", "PewCrypt   +decrypt"): ("EXCLUIR", "FALSO POSITIVO de subcadena: 'wcry' aparece dentro de Pe-WCRY-pt; familia sin relación"),
    ("WANNACRY", "ShadowCryptor"): ("EXCLUIR", "FALSO POSITIVO de subcadena: Shado-WCRY-ptor; sin relación"),
    ("WANNACRY", "Unknown Crypted"): ("EXCLUIR", "FALSO POSITIVO de subcadena: Unkno-WNCRY-pted; sin relación"),
    ("WASTEDLOCKER", "WastedLocker"): ("INCLUIR", "fn <encrypted_filename>_info: consistente con el hallazgo de nota-por-archivo; sin ext en MISP (son por víctima: .bbawasted...)"),
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def buscar_candidatas(valores):
    """Devuelve filas (familia, tipo_match, entrada) para las 30 familias."""
    por_value = {}
    for v in valores:
        por_value.setdefault(v["value"], v)
    filas = []
    for familia, alias in ALIAS.items():
        for v in valores:
            nombres = [v["value"]] + v.get("meta", {}).get("synonyms", [])
            nn = [norm(x) for x in nombres]
            tipo = None
            if any(a == x for a in alias for x in nn):
                tipo = "exacto"
            elif any(a in x for a in alias for x in nn):
                tipo = "subcadena"
            if tipo:
                filas.append((familia, tipo, v))
    # Controles que el alias no captura (p. ej. Crypt0L0cker con ceros): entran a mano.
    ya = {(f, v["value"]) for f, _, v in filas}
    for familia, value in EXTRAS:
        if (familia, value) in ya:
            continue
        if value in por_value:
            filas.append((familia, "manual", por_value[value]))
        else:
            print(f"[!] EXTRAS: la entrada '{value}' no está en este catálogo, se omite")
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--catalogo", type=Path, default=CATALOGO_DEF)
    ap.add_argument("--salida", type=Path, default=SALIDA_DEF)
    args = ap.parse_args()

    with open(args.catalogo, encoding="utf-8") as f:
        catalogo = json.load(f)
    valores = catalogo["values"]
    filas = buscar_candidatas(valores)
    args.salida.mkdir(parents=True, exist_ok=True)

    ruta_csv = args.salida / "mapeo_misp_borrador.csv"
    con_dictamen = sin_dictamen = 0
    resumen = {}  # familia -> dict con conteos tras aplicar el borrador
    with open(ruta_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["familia", "entrada_misp", "uuid", "tipo_match", "dictamen_borrador",
                    "motivo_borrador", "dictamen_romina", "synonyms",
                    "ransomnotes_filenames", "extensions", "n_textos_nota",
                    "fecha_misp", "primera_ref"])
        for familia, tipo, v in filas:
            m = v.get("meta", {})
            fn = m.get("ransomnotes-filenames", []) + m.get("ransomnotes-filesnames", [])
            dictamen, motivo = DICTAMENES.get(
                (familia, v["value"]), ("REVISAR", "SIN DICTAMEN: entrada nueva no revisada"))
            if (familia, v["value"]) in DICTAMENES:
                con_dictamen += 1
            else:
                sin_dictamen += 1
            w.writerow([familia, v["value"], v.get("uuid", ""), tipo, dictamen, motivo, "",
                        " | ".join(m.get("synonyms", [])), " | ".join(fn),
                        " | ".join(m.get("extensions", [])),
                        len(m.get("ransomnotes", [])), m.get("date", ""),
                        (m.get("refs") or [""])[0]])
            r = resumen.setdefault(familia, {"entradas": 0, "incluidas": 0,
                                             "revisar": 0, "fn": set(), "ext": set()})
            r["entradas"] += 1
            if dictamen == "INCLUIR":
                r["incluidas"] += 1
                r["fn"].update(fn)
                r["ext"].update(m.get("extensions", []))
            elif dictamen == "REVISAR":
                r["revisar"] += 1

    ruta_md = args.salida / "RESUMEN_mapeo.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Resumen del mapeo MISP (BORRADOR — auditar `dictamen_romina` en el CSV)\n\n")
        f.write(f"Catálogo: `{args.catalogo.name}` · candidatas: {len(filas)} · "
                f"con dictamen borrador: {con_dictamen} · sin dictamen: {sin_dictamen}\n\n")
        f.write("Cobertura POR FAMILIA contando SOLO las entradas con dictamen INCLUIR "
                "(cifras borrador, no citables hasta auditar):\n\n")
        f.write("| Familia | Entradas | Incluidas | A revisar | #fn | #ext |\n|---|---|---|---|---|---|\n")
        for familia in ALIAS:
            r = resumen.get(familia, {"entradas": 0, "incluidas": 0, "revisar": 0,
                                      "fn": set(), "ext": set()})
            f.write(f"| {familia} | {r['entradas']} | {r['incluidas']} | {r['revisar']} | "
                    f"{len(r['fn'])} | {len(r['ext'])} |\n")
        sin_entrada = [fam for fam in ALIAS if resumen.get(fam, {}).get("incluidas", 0) == 0]
        f.write(f"\n**Familias que quedan SIN entrada MISP propia tras el borrador:** "
                f"{', '.join(sin_entrada) if sin_entrada else 'ninguna'}.\n")
        f.write("\nCómo auditar: filtrar el CSV por `dictamen_borrador`, revisar cada fila "
                "(la columna `primera_ref` da la fuente de la entrada) y completar "
                "`dictamen_romina` con INCLUIR/EXCLUIR. Los REVISAR son decisiones de "
                "alcance para llevar al tutor.\n")

    print(f"Candidatas: {len(filas)} (dictamen borrador: {con_dictamen}, sin dictamen: {sin_dictamen})")
    print(f"CSV:     {ruta_csv}")
    print(f"Resumen: {ruta_md}")
    if sin_dictamen:
        print("[!] Hay entradas SIN dictamen (catálogo distinto del revisado el 2026-08-22): "
              "salen como REVISAR en el CSV.")


if __name__ == "__main__":
    main()
