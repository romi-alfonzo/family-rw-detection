#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capa_extension_cifrado.py -- la EXTENSION DE CIFRADO como una clave exacta mas de la cascada.

DE DONDE SALE. Hoy la cascada del frente de notas resuelve por regla exacta cuando la nota
trae un marcador (ONION, EMAIL, BTC, URL, ID, CLAVE -- la lista PATRONES de
normalizacion_marcadores.py) que en entrenamiento se vio asociado a UNA sola familia. Esa
lista NO incluye la extension que el ransomware le pone a los archivos que cifra, aunque casi
siempre esta escrita en el texto de la nota (".locked", ".cerber", ".LOCKBIT", "[.]encrypted",
".id-XXXX.[mail].casper"). Hoy esa senal solo entra de forma implicita, diluida en el TF-IDF de
caracteres junto con el resto del texto.

POR QUE DEBERIA SERVIR. La extension que agrega el ransomware es una de las senales mas
especificas de familia que existe en el dominio. Como clave exacta tiene exactamente el mismo
caracter que un correo o una billetera: si ya se vio asociada a una unica familia, la
identifica. Por eso se la agrega al MISMO diccionario, con el MISMO filtro de genericos (se
descarta toda clave que en entrenamiento aparezca en mas de una familia) y la MISMA regla de
unanimidad (la regla decide solo si todas las claves de la nota apuntan a una sola familia).
No se cambia nada mas de la cascada.

RESPALDO EMPIRICO DE QUE LA EXTENSION ES *LA* PARTE ROBUSTA DEL NOMBRE. Viene del OTRO frente
del trabajo (archivos cifrados), y se cita solo como argumento de dominio: los dos frentes van
separados y ninguna cifra de archivos respalda una cifra de notas. El 2026-09-28, en
2_codigo/exp2g_nombre_robusto.py, se documento que mirar el nombre ENTERO del archivo hacia que
el modelo aprendiera como NapierOne bautizo sus archivos y no como renombra el ransomware: con
los jpg fuera del entrenamiento, sumar la forma completa del nombre hundio el macro-F1 de 0,8052
a 0,2167, porque los jpg de NapierOne llevan un "-fromweb" en la base que los documentos no
tienen (0001-doc.doc.avos2 frente a 0001-jpg-fromweb.jpg.avos2). El arreglo fue calcular los
rasgos SOLO sobre la extension final. O sea: dentro de un nombre de archivo, la extension que
agrega el ransomware es la parte que generaliza y el resto es artefacto de como se armo el
corpus.

VENTAJA PROPIA DE ESTE DISENO, que conviene declarar. Aca la extension se extrae del TEXTO de la
nota, no del nombre del archivo. Eso la vuelve inmune a ese problema de raiz: lo que el
ransomware escribe adentro de la nota no depende de como el curador guardo el archivo. El
riesgo simetrico si existe y hay que vigilarlo: si el patron capturara ademas la extension del
documento ORIGINAL (de "0001-doc.doc.avos2" la extension del ransomware es ".avos2", NO
".doc.avos2"), quedarian claves como ".doc" o ".jpg" compartidas entre familias, que no
significan nada. Por eso la ruta de doble extension captura el componente que viene DESPUES de
la extension benigna, nunca la benigna, y ademas doc/docx/jpg/pdf/... estan en la lista de
descarte. Verificado en la muestra manual: ninguna de las 8 extensiones extraidas es una
extension de documento, y las rutas de doble extension y de cadena dan CERO ocurrencias en este
corpus.

=============================================================================================
EL PATRON DE EXTRACCION (lo que decide el resultado) -- verificado a mano ANTES de medir
=============================================================================================
Primer intento (descartado): barrer todo ".token" del texto. Sobre este corpus da 135 tokens
distintos y ~97 % es basura: dominios y TLD (.com, .org, .top, .onion, .torproject, .wikipedia),
rutas de URL (.cab, .casa, .guide, .rip, .plus de "http://xxx.onion.cab/"), extensiones del
ARCHIVO DE NOTA (.hta 210 veces en CERBER, .html, .txt), ejecutables (.exe de
"!WannaDecryptor!.exe"), numeros de version (GANDCRAB "V5.0.4" -> .0, .4), direcciones IP
(CLOP "\\10.30.12.98" -> .30, .12, .98) y hasta finales de oracion sin espacio ("AUTRE.ALORS",
"instructions.Otherwise"). Una clave basura compartida entre familias arruina la capa, asi que
el patron se endurecio hasta que la lista quedo limpia.

El patron final tiene tres pasos:

(1) DESOFUSCACION. "[.]", "(.)", "{.}" -> "."; "hxxp://" -> "http://".

(2) BORRADO DE LO QUE NO ES EXTENSION. Se blanquean con los patrones ya declarados en
    normalizacion_marcadores.py, pero en ESTE orden: primero la URL CON ESPACIOS INYECTADOS,
    despues URL, ONION, EMAIL, BTC, CLAVE e ID. Los dos cambios de orden son necesarios y los
    dos salieron de una captura basura concreta:
    - URL antes que ONION. Con ONION primero (el orden de normalizacion_marcadores),
      "http://decrypttozxybarc.onion.cab/[removed]" pierde el host y deja ".cab" suelto, que el
      extractor tomaba como extension de CERBER. Igual ".casa", ".guide", ".rip", ".plus" en
      GANDCRAB.
    - URL espaciada antes que todo. TESLACRYPT escribe "https://en .wikipedia. org/wiki/AES"
      (espacios inyectados dentro del host); el patron [URL] canonico corta en el primer espacio
      y deja ".wikipedia" suelto, que el extractor devolvia como extension de TESLACRYPT en 2
      notas: exactamente la clave basura de familia unica que mas dano haria, porque el filtro
      de genericos no la ve y actuaria como firma. Se probo antes la alternativa de colapsar los
      espacios alrededor de TODO punto y se descarto: tambien pega "have .encrypted" y hace
      perder la mencion legitima.

(3) CUATRO RUTAS DE CAPTURA sobre el texto ya limpio, con TOKEN = [A-Za-z0-9][A-Za-z0-9_-]{1,15}:
    a) ANCLA CON PUNTO: la palabra extension / extensi[o]n / erweiterung / estensione / suffix /
       sufijo, un separador (":", "=", raya o al menos un espacio) y ".TOKEN".
       Ej.: GANDCRAB "files are encrypted and have the extension: .GDCB";
            "sont cryptes et ont l'extension: .GACMW";
            NETWALKER "All encrypted files for this computer has extension: .eebf08".
    b) ANCLA SIN PUNTO: igual pero sin el punto, porque hay familias que lo omiten.
       Ej.: SODINOKIBI "all files on your system has extension lgzcfcr."
    c) SUELTA: un ".TOKEN" que es un token entero del texto -- delante solo puede haber inicio
       de linea, espacio o comilla/parentesis, y detras solo espacio o puntuacion de cierre.
       Ej.: LORENZ, cuya nota empieza literalmente con ".sz40 [+] What happened? [+]".
       La restriccion de los dos lados es la que mata las rutas de URL y los finales de oracion.
    d) DOBLE EXTENSION y CADENA DE VARIAS PARTES: "archivo.docx.locked" y
       ".id-ABC123.[mail@x.com].casper" (de esta ultima se toma el ultimo componente, y se
       aplica ANTES de blanquear correos, porque el correo va dentro de la cadena). Se
       implementan porque son la forma canonica del dominio; en ESTE corpus dan CERO
       ocurrencias y se reporta asi.

(4) FILTRO DE GENERICOS DEL PATRON (ademas del filtro por familia de la cascada). Se descarta
    el token si mide menos de 3 o mas de 16 caracteres, si no tiene ninguna letra (mata IPs y
    numeros de version), o si esta en una lista DECLARADA de: extensiones de archivo de nota y
    de ofimatica/ejecutables (txt, hta, html, exe, doc, jpg, zip, onion, ...), marcadores de
    plantilla del corpus (ext, extension, rand, snip, removed, uid, ...) y palabras funcionales
    de los cinco idiomas del corpus (the, your, de, die, les, il, ...), que es lo unico que la
    ruta (b) puede confundir con un valor.

RESULTADO DE LA VERIFICACION MANUAL (149 notas, corpus_v2): 8 extensiones distintas en 12 de
las 149 notas (8,05 %), CERO capturas basura. La lista completa, auditada una por una:
    .gacmw   GANDCRAB   (4 notas)   .gdcb    GANDCRAB   (2 notas)
    .ibkfz   GANDCRAB   (1 nota)    .krab    GANDCRAB   (1 nota)
    .rfncw   GANDCRAB   (1 nota)    .eebf08  NETWALKER  (1 nota)
    .lgzcfcr SODINOKIBI (1 nota)    .sz40    LORENZ     (1 nota)
El script vuelve a imprimir esta tabla al correr; si no coincide, cambio el corpus o el patron.

EL TECHO DEL CORPUS, QUE HAY QUE DECIR ANTES DE MEDIR. Las notas de este corpus casi nunca
escriben su extension: la mayoria de las fuentes publican la nota con los datos variables
sustituidos ("[snip]", "[removed]", "${EXTENSION}" en BLACKCAT, "{EXT}" en SODINOKIBI). Esos
marcadores de plantilla NO se cuentan como extension -- no son un valor, y tomarlos como clave
seria capturar texto de la plantilla, que es justo lo que el TF-IDF de caracteres ya hace. Por
eso la capa solo puede tocar 12 de 149 notas, y menos todavia bajo P2bal: para que la regla
dispare, el MISMO valor tiene que estar en una nota de entrenamiento, y el corte es por
plantilla. De las 8 extensiones, solo dos aparecen en mas de una plantilla (.gacmw en los
grupos 72 y 75, .gdcb en los grupos 73 y 74; ambas de GANDCRAB); las otras seis viven en una
sola plantilla y por construccion NO pueden disparar nunca. El maximo alcanzable es 6 notas de
149 (4,03 %), y solo en las semillas en que el reparto separa esas plantillas.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28). Se reporta igual lo que falle.

E1. PUERTA DE ENTRADA. La cascada SIN la capa reproduce la cifra de cabecera de P2bal:
    macro-F1 0,7417 y exactitud 0,8123 (tolerancia 0,01, fuente 4_resultados/_log_p2bal_149.txt).
    Si no, ABORTA y no se reporta nada.
E2. EXTRACCION. 8 extensiones distintas en 12 de 149 notas, CERO claves basura y CERO
    extensiones de documento coladas (doc, jpg, pdf, ...), tal cual la tabla de arriba. Es la
    unica prediccion ya verificada a mano (el encargo pedia verificar el patron antes de medir);
    se deja por escrito para que la corrida la confirme.
E3. FILTRO DE GENERICOS. CERO claves de extension descartadas por aparecer en mas de una
    familia en entrenamiento: ninguna de las 8 se repite entre familias. Si sale > 0, hay una
    colision que no vi.
E4. COBERTURA DE LA CAPA (primera columna). Entre 0,010 y 0,040 de las notas. El techo duro es
    6/149 = 0,0403 y se alcanza solo si el reparto separa siempre los grupos 72|75 y 73|74.
E5. ACIERTO DONDE APLICA (segunda columna). EXACTAMENTE 1,000. Las dos unicas claves que pueden
    disparar (.gacmw, .gdcb) son de GANDCRAB y de nadie mas, asi que la capa no puede
    equivocarse donde aplica. Si sale < 1,000 hay un error de implementacion.
E6. LA PREGUNTA PRINCIPAL (tercera columna). Delta macro-F1 pareado por semilla contra la
    cascada actual: se predice |Delta| <= 0,002, IC 95 % que INCLUYE el cero, y menos de 10 de
    las 50 semillas con Delta estrictamente positivo. Razon: las unicas notas que la capa puede
    alcanzar son notas de GANDCRAB, una familia con 4 plantillas y texto muy marcado
    ("GANDCRAB V5.0"), que la cascada ya resuelve casi siempre sin ayuda.
E7. TECHO ORACULO. Aun regalandole a la capa la respuesta correcta en las 12 notas que SI traen
    extension (oraculo imposible en la practica), la ganancia de macro-F1 es <= +0,030. Es la
    cota superior de la tecnica sobre este corpus y separa "la idea no sirve" de "la idea no
    tiene material en este corpus".
E8. INTERPRETACION FIJADA DE ANTEMANO, para no racionalizar despues:
    - Si E6 falla porque el IC excluye el cero por arriba: la extension aporta y se adopta.
    - Si E6 se cumple (IC incluye el cero) Y E7 se cumple: la capa no aporta, y la causa es el
      CORPUS, no el metodo -- las notas publicadas no traen la extension. Es un resultado
      negativo informativo, y se reporta como limitacion del corpus, con el numero del techo.
    - Si E6 se cumple pero E7 falla (el oraculo si ganaria mas de 0,030): entonces el material
      esta y lo que falla es el emparejamiento por valor exacto bajo corte por plantilla; la
      linea a seguir seria generalizar la clave, no descartarla.

NOTA DE HONESTIDAD. El patron se ajusto mirando este corpus (dos iteraciones: agregar el
borrado de URL antes que ONION, y deshacer el espaciado inyectado). Por lo tanto la cifra de
extraccion NO es una validacion independiente del patron: mide que el patron, AJUSTADO a este
corpus, no captura basura. Sobre notas nuevas habria que volver a auditarlo. Se declara asi.
=============================================================================================

Uso:  python capa_extension_cifrado.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from normalizacion_marcadores import PATRONES
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_capa_extension_149"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01

# Lo que la verificacion manual dejo fijado (E2). Se vuelve a comprobar al correr.
ESPERADO_DISTINTAS, ESPERADO_NOTAS = 8, 12

# ============================================================
# EL PATRON DE EXTRACCION (ver el bloque del docstring)
# ============================================================
_P = dict(PATRONES)
# URL primero a proposito: tiene que llevarse la URL entera antes de que ONION le saque el host
# y deje la ruta suelta (".cab", ".casa", ".guide" de "http://xxx.onion.cab/").
_ORDEN_BORRADO = ("[URL]", "[ONION]", "[EMAIL]", "[BTC]", "[CLAVE]", "[ID]")

# Extensiones que NO son de cifrado: archivo de nota, ofimatica, ejecutables, red.
_STOP_ARCHIVO = {
    "txt", "hta", "html", "htm", "xhtml", "php", "asp", "aspx", "jsp", "css", "js", "json",
    "xml", "md", "exe", "dll", "bat", "cmd", "ps1", "vbs", "jar", "msi", "sys", "com", "scr",
    "bin", "tmp", "log", "ini", "dat", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "pdf",
    "rtf", "odt", "ods", "csv", "mdb", "accdb", "pst", "jpg", "jpeg", "png", "gif", "bmp",
    "tiff", "svg", "ico", "mp3", "mp4", "avi", "mkv", "wav", "zip", "rar", "7z", "tar", "gz",
    "bz2", "iso", "url", "lnk", "reg", "key", "onion", "oni0n", "i2p", "tor",
}
# Marcadores de plantilla con que las fuentes publican las notas saneadas.
_STOP_PLANTILLA = {
    "ext", "extension", "extensions", "rand", "random", "snip", "removed", "enter", "connect",
    "recommended", "uid", "id", "victim", "xxx", "xxxx", "yyy", "abc", "name", "file", "files",
}
# Palabras funcionales de los cinco idiomas del corpus: lo unico que la ruta SIN punto puede
# confundir con un valor de extension.
_STOP_PALABRA = {
    "in", "of", "the", "to", "and", "is", "are", "for", "your", "all", "an", "that", "it", "we",
    "you", "this", "these", "those", "by", "with", "as", "no", "not", "can", "will", "was",
    "were", "be", "been", "have", "has", "if", "or", "on", "at", "from", "after", "before",
    "now", "so", "but", "do", "does", "don", "its", "our", "their", "one", "they", "them",
    "es", "de", "la", "el", "los", "las", "una", "que", "con", "por", "para", "del", "se",
    "su", "sus", "son", "como", "todos", "sus",
    "die", "der", "das", "und", "ist", "mit", "fur", "den", "dem", "ein", "eine", "sie",
    "ihre", "auf", "nicht", "alle", "haben", "wurden",
    "les", "des", "du", "et", "est", "pour", "vos", "votre", "sur", "dans", "une", "sont",
    "ont", "tous", "avec", "fichiers",
    "di", "il", "le", "lo", "gli", "non", "sono", "che", "per", "dei", "della", "tutti",
    "vostri",
}
STOP_EXT = _STOP_ARCHIVO | _STOP_PLANTILLA | _STOP_PALABRA

_TOK = r"[A-Za-z0-9][A-Za-z0-9_\-]{1,15}"
_BENIGNA = (r"(?:doc|docx|xls|xlsx|ppt|pptx|pdf|jpg|jpeg|png|gif|bmp|txt|csv|zip|rar|mdb|sql|"
            r"bak|dbf|psd|mp3|mp4|avi|dwg|rtf|odt|accdb|pst|7z|tar|gz)")
# "extension" en los idiomas del corpus. La o acentuada va escapada: no se meten tildes en el codigo.
_ANCLA = r"(?:extensi[o\u00f3]n(?:es)?|extensions?|erweiterung|estensione|suffix|sufijo)"
# Separador: signo de valor, o al menos un espacio. Exigir uno de los dos evita que, despues de
# deshacer el espaciado inyectado, "extension.Alle" se lea como la extension ".alle".
_SEP = (r"(?:[ \t]*[:=\u2013\u2014][ \t]*|[ \t]+)"
        r"(?:(?:of|de|is|are|es|son)[ \t]+)?[\"'\u00ab\u201c\(\[]?")

# URL con espacios inyectados: TESLACRYPT escribe "https://en .wikipedia. org/wiki/AES".
RX_URL_ESPACIADA = re.compile(
    r"https?://(?:[\w\-]+[ \t]*\.[ \t]*)+[\w\-]+(?:[ \t]*/[^\s]*)?", re.I)
RX_CADENA = re.compile(r"\.id[-_][A-Za-z0-9]{4,}(?:\.\[[^\]\s]{3,60}\])?\.(" + _TOK + r")", re.I)
RX_ANCLA_PUNTO = re.compile(_ANCLA + _SEP + r"\.(" + _TOK + r")", re.I)
RX_ANCLA_SIN_PUNTO = re.compile(_ANCLA + _SEP + r"(" + _TOK + r")", re.I)
RX_SUELTA = re.compile(r"(?:^|(?<=[\s\"'(\[\u00ab]))\.(" + _TOK + r")(?=[\s\"'.,;:!?)\]\u00bb]|$)",
                       re.M)
RX_DOBLE = re.compile(r"\.(?:" + _BENIGNA + r")\.(" + _TOK + r")\b", re.I)


def desofuscar(texto):
    """Deshace las ofuscaciones del corpus que rompen la extraccion."""
    t = re.sub(r"\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}", ".", texto)
    t = re.sub(r"h(?:xx|\*\*)p(s?)\s*://", r"http\1://", t, flags=re.I)
    return t


def _blanquear(texto):
    # Primero la URL con espacios inyectados: el patron [URL] canonico corta en el primer
    # espacio y dejaria ".wikipedia" suelto. Se borra el host espaciado ENTERO, sin tocar el
    # resto del texto: colapsar los espacios alrededor de TODO punto se probo y es peor, porque
    # tambien pega "have .encrypted" y se pierde la mencion legitima.
    t = RX_URL_ESPACIADA.sub(" ", texto)
    for etiqueta in _ORDEN_BORRADO:
        t = _P[etiqueta].sub(" ", t)
    return t


def _valido(v):
    v = v.lower().strip("-_.")
    if not (3 <= len(v) <= 16):
        return None
    if not re.search(r"[a-z]", v):        # mata IPs y numeros de version
        return None
    if v in STOP_EXT:
        return None
    return v


def extraer_extensiones(texto, con_detalle=False):
    """Devuelve el conjunto de extensiones de cifrado mencionadas en la nota.

    Con con_detalle=True devuelve {extension: [(ruta, fragmento), ...]} para auditar a mano.
    """
    hallazgos = defaultdict(list)
    t = desofuscar(texto)
    # la cadena de varias partes se busca ANTES de blanquear: lleva un correo adentro
    for m in RX_CADENA.finditer(t):
        v = _valido(m.group(1))
        if v:
            hallazgos[v].append(("cadena", m.group(0)[:60]))
    tl = _blanquear(t)
    for etiqueta, rx in (("ancla_punto", RX_ANCLA_PUNTO),
                         ("ancla_sin_punto", RX_ANCLA_SIN_PUNTO),
                         ("suelta", RX_SUELTA),
                         ("doble", RX_DOBLE)):
        for m in rx.finditer(tl):
            v = _valido(m.group(1))
            if v:
                hallazgos[v].append((etiqueta, m.group(0)[:60]))
    return hallazgos if con_detalle else set(hallazgos)


# ============================================================
# La capa: una clave exacta MAS, en el mismo diccionario
# ============================================================
def dicc_con_extension(tr, iocs, nombres_nota, exts, y):
    """Como protocolo_logo.dicc_privados, pero sumando las claves ("[EXT]", valor).

    Mismo filtro de genericos: se borra toda clave que en entrenamiento apunte a mas de una
    familia. Devuelve (diccionario, descartadas_ext, total_claves_ext).
    """
    d = defaultdict(set)
    for i in tr:
        for c in iocs[i]:
            d[c].add(y[i])
        if nombres_nota[i]:
            d[("[NOMBRE]", nombres_nota[i])].add(y[i])
        for v in exts[i]:
            d[("[EXT]", v)].add(y[i])
    total_ext = sum(1 for k in d if k[0] == "[EXT]")
    genericas = [k for k, v in d.items() if len(v) > 1]
    desc_ext = sum(1 for k in genericas if k[0] == "[EXT]")
    for k in genericas:
        del d[k]
    return d, desc_ext, total_ext


def regla_con_extension(i, d, iocs, nombres_nota, exts):
    """Como protocolo_logo.regla, con las claves de extension sumadas al mismo conjunto."""
    claves = set(iocs[i])
    if nombres_nota[i]:
        claves.add(("[NOMBRE]", nombres_nota[i]))
    claves |= {("[EXT]", v) for v in exts[i]}
    fams = set()
    for c in claves:
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


def regla_solo_extension(i, d, exts):
    """La capa SOLA, para medir su cobertura y su acierto donde aplica."""
    fams = set()
    for v in exts[i]:
        c = ("[EXT]", v)
        if c in d:
            fams |= d[c]
    return next(iter(fams)) if len(fams) == 1 else None


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
    S = args.n_semillas

    print("=" * 90)
    print("  LA EXTENSION DE CIFRADO DE LA NOTA COMO UNA CLAVE EXACTA MAS DE LA CASCADA")
    print("=" * 90)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))} | Semillas: {S}")

    # ---------------------------------------------------------
    # VERIFICACION MANUAL DEL PATRON (antes de medir nada)
    # ---------------------------------------------------------
    detalles = [extraer_extensiones(t, con_detalle=True) for t in textos]
    exts = [set(dd) for dd in detalles]
    fam_de_ext = defaultdict(set)
    grupos_de_ext = defaultdict(set)
    rutas = Counter()
    filas_ext = []
    for i in range(n):
        for v, ms in detalles[i].items():
            fam_de_ext[v].add(y[i])
            grupos_de_ext[v].add(int(grupos[i]))
            for r, _ in ms:
                rutas[r] += 1
            filas_ext.append(dict(familia=y[i], archivo=archivos[i], grupo=int(grupos[i]),
                                  extension=v, ruta=ms[0][0], fragmento=ms[0][1]))
    n_con = sum(1 for e in exts if e)
    print("\n" + "-" * 90)
    print("  VERIFICACION MANUAL DEL PATRON (E2) -- si esto no cuadra, no se sigue")
    print("-" * 90)
    print(f"  extensiones distintas : {len(fam_de_ext)}  (esperado {ESPERADO_DISTINTAS})")
    print(f"  notas con >= 1        : {n_con}/{n} = {n_con/n:.4f}  (esperado {ESPERADO_NOTAS})")
    print(f"  ocurrencias por ruta  : {dict(rutas)}")
    print(f"\n  {'extension':<12}{'familias':>9}  {'notas':>5}  {'plantillas':>10}  familia(s)")
    for v in sorted(fam_de_ext):
        nn = sum(1 for e in exts if v in e)
        print(f"  .{v:<11}{len(fam_de_ext[v]):>9}  {nn:>5}  "
              f"{str(sorted(grupos_de_ext[v])):>10}  {sorted(fam_de_ext[v])}")
    dfe = pd.DataFrame(filas_ext).sort_values(["familia", "archivo"])
    dfe.to_csv(OUT / "extension_extracciones.csv", index=False, encoding="utf-8-sig")
    print("\n  detalle nota a nota (las 20 primeras; el CSV trae todas):")
    print(dfe.head(20).to_string(index=False))
    # cuantas pueden disparar alguna vez: las que viven en >= 2 plantillas
    pueden = [v for v in fam_de_ext if len(grupos_de_ext[v]) >= 2]
    notas_alcanzables = sum(1 for e in exts if any(v in pueden for v in e))
    print(f"\n  extensiones presentes en >= 2 plantillas (unicas que pueden disparar bajo corte "
          f"por plantilla): {sorted(pueden)}")
    print(f"  TECHO DURO de cobertura: {notas_alcanzables}/{n} = {notas_alcanzables/n:.4f}")
    compartidas = [v for v, fs in fam_de_ext.items() if len(fs) > 1]
    print(f"  extensiones compartidas entre familias (basura potencial): "
          f"{sorted(compartidas) if compartidas else 'ninguna'}")
    # control del riesgo que dejo a la vista el frente de archivos (exp2g): que se cuele la
    # extension del documento ORIGINAL ("0001-doc.doc.avos2" -> la del ransomware es .avos2).
    documentales = [v for v in fam_de_ext if v in _STOP_ARCHIVO]
    print(f"  extensiones de DOCUMENTO coladas (deberia ser ninguna): "
          f"{sorted(documentales) if documentales else 'ninguna'}  | "
          f"rutas doble+cadena: {rutas['doble'] + rutas['cadena']} ocurrencias")
    e2 = (len(fam_de_ext) == ESPERADO_DISTINTAS and n_con == ESPERADO_NOTAS
          and not documentales)

    # ---------------------------------------------------------
    # EVALUACION
    # ---------------------------------------------------------
    p_base = np.empty((S, n), dtype=object)     # cascada actual
    p_ext = np.empty((S, n), dtype=object)      # cascada + capa de extension
    aplica_base = np.zeros((S, n), dtype=bool)  # cobertura de la regla de siempre
    aplica_ext = np.zeros((S, n), dtype=bool)   # cobertura de la capa NUEVA, sola
    ok_ext = np.zeros((S, n), dtype=bool)       # acierto de la capa nueva donde aplica
    desc_gen, tot_claves_ext = [], []

    print("\nEvaluando ...")
    for s in range(S):
        rng = np.random.default_rng(20_000 + s)
        for tr, te in split_p2bal(y, grupos, familias, rng):
            vec = vectorizador("combinado")
            Xtr = vec.fit_transform(textos_arr[tr])
            Xte = vec.transform(textos_arr[te])
            clf = obtener_modelos(s)["LinearSVC"]
            clf.fit(Xtr, y[tr])
            pt = clf.predict(Xte)

            d0 = dicc_privados(tr, iocs, nombres_nota, y)
            d1, desc, tot = dicc_con_extension(tr, iocs, nombres_nota, exts, y)
            desc_gen.append(desc)
            tot_claves_ext.append(tot)

            for k, i in enumerate(te):
                r0 = regla(i, d0, iocs, nombres_nota)
                p_base[s, i] = pt[k] if r0 is None else r0
                aplica_base[s, i] = r0 is not None

                r1 = regla_con_extension(i, d1, iocs, nombres_nota, exts)
                p_ext[s, i] = pt[k] if r1 is None else r1

                rs = regla_solo_extension(i, d1, exts)
                if rs is not None:
                    aplica_ext[s, i] = True
                    ok_ext[s, i] = (rs == y[i])
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{S} semillas")

    # ---------------- puerta E1 ----------------
    f1_ba = np.array([f1_score(y, p_base[s], average="macro", labels=familias, zero_division=0)
                      for s in range(S)])
    ac_ba = np.array([accuracy_score(y, p_base[s]) for s in range(S)])
    print("\n" + "-" * 90)
    print("  E1 -- PUERTA DE ENTRADA (la cascada SIN la capa debe reproducir la cabecera)")
    print("-" * 90)
    print(f"  base: macro-F1 {f1_ba.mean():.4f} vs {CANON_F1} | "
          f"exactitud {ac_ba.mean():.4f} vs {CANON_ACC}")
    ok_puerta = abs(f1_ba.mean() - CANON_F1) <= TOL and abs(ac_ba.mean() - CANON_ACC) <= TOL
    if not ok_puerta and not args.sin_puerta:
        sys.exit("ABORTADO (E1): la cascada base no reproduce la cabecera. No se reporta nada.")
    print("  OK\n" if ok_puerta else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------------- metricas ----------------
    f1_ex = np.array([f1_score(y, p_ext[s], average="macro", labels=familias, zero_division=0)
                      for s in range(S)])
    ac_ex = np.array([accuracy_score(y, p_ext[s]) for s in range(S)])

    # oraculo: se le regala la respuesta en TODA nota que trae extension (techo de la tecnica)
    p_ora = p_base.copy()
    idx_con_ext = [i for i in range(n) if exts[i]]
    for s in range(S):
        for i in idx_con_ext:
            p_ora[s, i] = y[i]
    f1_or = np.array([f1_score(y, p_ora[s], average="macro", labels=familias, zero_division=0)
                      for s in range(S)])
    ac_or = np.array([accuracy_score(y, p_ora[s]) for s in range(S)])

    filas = []
    for etq, pm, f1v, acv in (("cascada actual (sin la capa)", p_base, f1_ba, ac_ba),
                              ("cascada + capa de extension", p_ext, f1_ex, ac_ex),
                              ("ORACULO sobre las notas con extension (techo)", p_ora, f1_or, ac_or)):
        filas.append(dict(sistema=etq,
                          macro_f1=round(float(f1v.mean()), 4),
                          exactitud=round(float(acv.mean()), 4),
                          bal=round(float(np.mean([balanced_accuracy_score(y, pm[s])
                                                   for s in range(S)])), 4),
                          mcc=round(float(np.mean([matthews_corrcoef(y, pm[s])
                                                   for s in range(S)])), 4)))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "extension_resumen.csv", index=False, encoding="utf-8-sig")
    print("=== RESULTADO (149 notas, 30 familias, P2bal, 50 semillas) ===")
    print(df.to_string(index=False))

    # ---------------- LAS TRES COLUMNAS ----------------
    cob_ext = np.array([aplica_ext[s].mean() for s in range(S)])
    acierto_ext = np.array([ok_ext[s][aplica_ext[s]].mean() if aplica_ext[s].any() else np.nan
                            for s in range(S)])
    cob_base = np.array([aplica_base[s].mean() for s in range(S)])
    d_f1 = f1_ex - f1_ba
    d_ac = ac_ex - ac_ba
    mf, lof, hif = ic(d_f1)
    ma, loa, hia = ic(d_ac)
    n_dec = int(aplica_ext.sum())
    n_ok = int(ok_ext.sum())
    acierto_global = (n_ok / n_dec) if n_dec else float("nan")

    tres = pd.DataFrame([dict(
        capa="extension de cifrado",
        cobertura_de_la_capa=round(float(cob_ext.mean()), 4),
        cobertura_ic95=f"[{ic(cob_ext)[1]:.4f}; {ic(cob_ext)[2]:.4f}]",
        decisiones_totales=n_dec,
        acierto_donde_aplica=round(acierto_global, 4) if n_dec else np.nan,
        efecto_macro_f1=round(float(mf), 4),
        efecto_macro_f1_ic95=f"[{lof:+.4f}; {hif:+.4f}]",
        semillas_positivas=f"{int((d_f1 > 0).sum())}/{S}")])
    tres.to_csv(OUT / "extension_tres_columnas.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 90)
    print("  LAS TRES COLUMNAS (nunca una sola)")
    print("=" * 90)
    print(f"  1) COBERTURA DE LA CAPA          : {cob_ext.mean():.4f} "
          f"[{ic(cob_ext)[1]:.4f}; {ic(cob_ext)[2]:.4f}]  "
          f"({n_dec} decisiones en {S} semillas x {n} notas; "
          f"techo duro {notas_alcanzables/n:.4f})")
    print(f"  2) ACIERTO DONDE APLICA          : "
          f"{acierto_global:.4f} ({n_ok}/{n_dec})" if n_dec
          else "  2) ACIERTO DONDE APLICA          : no aplica nunca (0 decisiones)")
    print(f"  3) EFECTO SOBRE EL MACRO-F1      : {mf:+.4f} [{lof:+.4f}; {hif:+.4f}]  "
          f"{int((d_f1 > 0).sum())}/{S} semillas positivas")
    print(f"     (efecto sobre la exactitud    : {ma:+.4f} [{loa:+.4f}; {hia:+.4f}]  "
          f"{int((d_ac > 0).sum())}/{S} semillas positivas)")
    print(f"\n  Contexto: cobertura de la regla de siempre (IOC + nombre) {cob_base.mean():.4f}; "
          f"la capa nueva agrega {cob_ext.mean():.4f}.")
    print(f"  Claves ('[EXT]', valor) en el diccionario de entrenamiento: "
          f"{np.mean(tot_claves_ext):.2f} por pliegue.")
    print(f"  DESCARTADAS POR EL FILTRO DE GENERICOS (en > 1 familia en entrenamiento): "
          f"{np.mean(desc_gen):.2f} por pliegue, {int(np.sum(desc_gen))} en total.")

    # ---------------- donde cambia la respuesta ----------------
    cambia = (p_base != p_ext)
    mejora = cambia & (p_ext == y[None, :]) & (p_base != y[None, :])
    empeora = cambia & (p_base == y[None, :]) & (p_ext != y[None, :])
    print(f"\n  La capa CAMBIA la respuesta en {int(cambia.sum())} de {S*n} decisiones "
          f"({cambia.sum()/(S*n):.4f}): mejora {int(mejora.sum())}, empeora {int(empeora.sum())}, "
          f"neutra {int(cambia.sum() - mejora.sum() - empeora.sum())}.")
    if cambia.any():
        det = Counter()
        for s in range(S):
            for i in np.where(cambia[s])[0]:
                det[(y[i], str(p_base[s, i]), str(p_ext[s, i]))] += 1
        dc = pd.DataFrame([dict(familia_real=a, sin_capa=b, con_capa=c, veces=k)
                           for (a, b, c), k in det.most_common()])
        dc.to_csv(OUT / "extension_cambios.csv", index=False, encoding="utf-8-sig")
        print(dc.head(15).to_string(index=False))

    # ---------------- por familia ----------------
    ffilas = []
    for f in familias:
        idx = np.where(y == f)[0]
        ffilas.append(dict(
            familia=f, n_notas=len(idx),
            trae_extension=int(sum(1 for i in idx if exts[i])),
            base=round(float(np.mean([(p_base[s, idx] == f).mean() for s in range(S)])), 4),
            con_capa=round(float(np.mean([(p_ext[s, idx] == f).mean() for s in range(S)])), 4)))
    dff = pd.DataFrame(ffilas)
    dff["delta"] = (dff.con_capa - dff.base).round(4)
    dff.to_csv(OUT / "extension_por_familia.csv", index=False, encoding="utf-8-sig")
    print("\n=== POR FAMILIA (solo las que traen alguna extension en el texto) ===")
    print(dff[dff.trae_extension > 0].to_string(index=False))

    # ---------------- veredicto ----------------
    d_or = f1_or - f1_ba
    mo, loo, hio = ic(d_or)
    e3 = int(np.sum(desc_gen)) == 0
    e4 = 0.010 <= cob_ext.mean() <= 0.040
    e5 = (n_dec > 0) and abs(acierto_global - 1.0) < 1e-12
    e6 = (abs(mf) <= 0.002) and (lof <= 0 <= hif) and (int((d_f1 > 0).sum()) < 10)
    e7 = mo <= 0.030
    print("\n" + "=" * 90)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 90)
    chk = [
        ("E1 puerta de entrada", ok_puerta, f"{f1_ba.mean():.4f} / {ac_ba.mean():.4f}"),
        ("E2 extraccion: 8 extensiones en 12 notas, sin basura", e2,
         f"{len(fam_de_ext)} extensiones, {n_con} notas, "
         f"compartidas {len(compartidas)}, documentales {len(documentales)}"),
        ("E3 cero claves descartadas por generico", e3, f"{int(np.sum(desc_gen))}"),
        ("E4 cobertura de la capa entre 0,010 y 0,040", e4, f"{cob_ext.mean():.4f}"),
        ("E5 acierto donde aplica = 1,000", e5,
         f"{acierto_global:.4f} ({n_ok}/{n_dec})" if n_dec else "0 decisiones"),
        ("E6 |Delta macro-F1| <= 0,002, IC incluye 0, < 10/50 semillas +", e6,
         f"D {mf:+.4f} [{lof:+.4f}; {hif:+.4f}] {int((d_f1 > 0).sum())}/{S}"),
        ("E7 techo oraculo <= +0,030 de macro-F1", e7,
         f"D {mo:+.4f} [{loo:+.4f}; {hio:+.4f}]"),
    ]
    for nombre, cumple, det_ in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<62} {det_}")

    print("\n  LECTURA SEGUN E8 (fijada antes de correr):")
    if lof > 0:
        print("    La capa APORTA: el IC del Delta excluye el cero por arriba. Se adopta la")
        print("    extension de cifrado como clave exacta de la cascada.")
    elif e7:
        print("    La capa NO aporta, y la causa es el CORPUS, no el metodo: las notas publicadas")
        print("    casi nunca escriben su extension (12 de 149 la traen, y solo 2 valores viven en")
        print("    mas de una plantilla, que es lo unico que puede disparar bajo corte por")
        print("    plantilla). Incluso el oraculo, que acierta en las 12, sube el macro-F1")
        print(f"    apenas {mo:+.4f}. Es un resultado negativo informativo: se reporta como")
        print("    limitacion del corpus, con el techo pegado.")
    else:
        print("    La capa NO aporta PERO el oraculo si ganaria: el material esta en el texto y lo")
        print("    que falla es el emparejamiento por VALOR EXACTO bajo corte por plantilla. La")
        print("    linea a seguir seria generalizar la clave, no descartarla.")

    resumen = dict(
        n_notas=n, n_familias=len(familias), n_semillas=S,
        extensiones_distintas=len(fam_de_ext), notas_con_extension=n_con,
        extensiones_compartidas_entre_familias=len(compartidas),
        techo_duro_cobertura=round(notas_alcanzables / n, 4),
        cobertura_capa=round(float(cob_ext.mean()), 4),
        acierto_donde_aplica=round(acierto_global, 4) if n_dec else None,
        delta_macro_f1=round(float(mf), 4), delta_ic95=[round(lof, 4), round(hif, 4)],
        semillas_positivas=int((d_f1 > 0).sum()),
        delta_oraculo=round(float(mo), 4),
        claves_descartadas_genericas=int(np.sum(desc_gen)))
    pd.DataFrame([resumen]).to_csv(OUT / "extension_manifiesto.csv", index=False,
                                   encoding="utf-8-sig")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
