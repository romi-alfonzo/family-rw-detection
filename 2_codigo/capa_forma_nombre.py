#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capa_forma_nombre.py -- ¿la FORMA del nombre de la nota sirve como capa de la cascada?

DE DONDE SALE: ES UNA TRANSFERENCIA DESDE EL FRENTE DE ARCHIVOS.
En el Exp. 2d (frente de archivos, job 3937, cerrado el 2026-09-17) la forma del nombre del
archivo cifrado resulto DECISIVA al combinarla con los bytes:
    solo bytes .................... macro-F1 0,9117
    bytes + forma del nombre ...... macro-F1 0,9998   (Delta +0,0880, IC95 [+0,0866; +0,0894], 5/5)
    solo forma del nombre ......... macro-F1 0,5771
La forma SOLA no resuelve la tarea (0,577): COMPLEMENTA en vez de reemplazar. Los bytes y la
forma aciertan cosas distintas. Ese hallazgo NUNCA se probo en el frente de notas, y los dos
frentes van separados, asi que aca hay que medirlo de nuevo con las cifras propias de notas.

QUE USA HOY LA CASCADA Y QUE NO. La cascada de notas ya usa el nombre genuino del archivo de la
nota, pero como CLAVE EXACTA: si en entrenamiento vio «info.hta» asociado a una sola familia, la
usa. Lo que NO usa es la FORMA del nombre: su patron. «HOW_TO_DECRYPT.txt»,
«HOW-TO-DECRYPT.txt» y «HOW TO DECRYPT.txt» son tres claves exactas distintas y una sola forma.
En el corpus pasa exactamente eso: CERBER tiene «# decrypt my files #.txt» y
«# decrypt_my_files #.txt» (dos claves, una forma), y siete variantes de «_..._[]_.hta».

=============================================================================================
LAS CINCO ABSTRACCIONES QUE SE PRUEBAN, Y POR QUE CADA UNA

Todas se calculan sobre el nombre ya normalizado por cargar_nombres() (minusculas, y el ID de
la victima ya enmascarado como «[]» por la auditoria). Todas parten el nombre en base +
extension por el ULTIMO punto.

  ESQ   ESQUELETO TIPOGRAFICO. En la base, cada corrida de letras -> «W», cada corrida de
        digitos -> «D», el ID -> «[]», y TODO lo demas (separadores, signos, adornos) queda
        literal; se le pega la extension. Borra las palabras y conserva el andamiaje.
        «_read_this_file_[]_.txt» -> «_W_W_W_[]_.txt».
        POR QUE: es el analogo directo de la «forma» del Exp. 2d. Si una familia tiene un
        molde de nombre propio (los adornos «#...#», «!!...!!», «_..._[]_»), el esqueleto lo
        captura aunque cambien las palabras.

  ESQC  ESQUELETO COLAPSADO. El ESQ, pero colapsando toda repeticion «W<sep>W<sep>W» en «W+»
        para cada separador (_, -, espacio). «_W_W_W_[]_.txt» y «_W_W_[]_.txt» -> «_W+_[]_.txt».
        POR QUE: el ESQ separa por CANTIDAD de palabras, que es justo lo que mas varia entre
        notas de una misma familia («_help_decrypt_[]_» contra «_read_this_file_[]_», las dos
        de CERBER). Colapsar deberia generalizar dentro de la familia. El riesgo es el
        contrario y hay que medirlo: colapsar tambien borra la diferencia entre familias.

  FIRMA FIRMA ESTRUCTURAL: la tupla (extension · separador dominante · cantidad de tokens en
        tramos 0/1/2/3/4+ · tiene digitos · tiene ID de victima · largo de la base en tramos
        corto/medio/largo/muylargo). Es la abstraccion mas gruesa y es la mas parecida a lo que
        el Exp. 2d le dio al bosque: pocas columnas densas, ninguna palabra.
        POR QUE: prueba si la senal esta en la GEOMETRIA del nombre y no en su contenido. Es la
        traduccion mas literal del Exp. 2d, donde el clasificador nunca vio la palabra.

  DOM   PALABRAS DEL DOMINIO + EXTENSION. Se borra el ID, se quitan TODOS los caracteres no
        alfanumericos y sobre la cadena resultante se marca que palabras de un vocabulario fijo
        aparecen como subcadena; el valor es el conjunto ordenado + la extension.
        «# decrypt my files #.txt» y «# decrypt_my_files #.txt» -> «crypt+decrypt+file+files+my.txt».
        POR QUE: es la abstraccion COMPLEMENTARIA de las tres anteriores: tira el andamiaje y se
        queda con el contenido. Al borrar separadores y adornos unifica las variantes de
        puntuacion, que es el caso que motivo todo el experimento. Si la senal del nombre es
        lexica y no geometrica, esta es la que tiene que ganar.

  EXT   SOLO LA EXTENSION (.txt / .hta / .html / .htm). CONTROL, no candidata.
        POR QUE: en el Exp. 2d la extension literal sola dio 0,9244, mas que los bytes solos;
        habia que descartar que todo el aporte fuera «es un .hta». Aca ademas hace de control
        del FILTRO DE GENERICOS: .txt, .html y .hta aparecen en muchas familias y el filtro de
        unanimidad los tiene que borrar del diccionario. Si EXT mueve el agregado, el filtro no
        esta funcionando y el resto de las cifras no son confiables.

  TODAS ESQ + ESQC + FIRMA + DOM a la vez, fijado ANTES de correr (no se eligen ganadoras).
  SELECCION Las que individualmente den Delta medio > 0. Se declara POST-HOC y se reporta como tal.

COMO SE INTEGRAN: cada abstraccion entra como UNA CLAVE EXACTA MAS en el MISMO diccionario de
la cascada, con el MISMO filtro de genericos y la MISMA regla de unanimidad que los IOCs y que
el nombre exacto: una clave que en entrenamiento apunta a dos familias se borra, y en prueba la
regla responde solo si TODAS las claves que disparan coinciden en una sola familia. El nombre
exacto NO se saca: la forma se AGREGA. Por eso el efecto puede ser NEGATIVO y hay que medirlo:
una clave de forma nueva puede romper la unanimidad de un nombre exacto que acertaba.

=============================================================================================
LA LIMITACION, DECLARADA ANTES DE CUALQUIER CIFRA

Solo 64 de las 149 notas tienen nombre genuino auditado (42,95 %). Las otras 85 fueron
renombradas por quien las recolecto y usar esos nombres seria CIRCULAR. Se trabaja unicamente
con los 64 auditados (los que cargar_nombres() devuelve; el resto llega como None).

  >>> TECHO ESTRUCTURAL: ninguna capa basada en el nombre puede cubrir mas del 42,95 % del
  >>> corpus. Ese techo NO es del metodo: es del corpus, y no se arregla con mejores
  >>> abstracciones. Toda cifra global de este script esta multiplicada por 64/149.

Por eso TODO se reporta dos veces: GLOBAL (149 notas) y RESTRINGIDO (las 64 con nombre). Son
dos cifras distintas y las dos importan: la global dice cuanto mueve la tesis hoy, la
restringida dice cuanto moveria si el corpus estuviera auditado entero.

Segunda limitacion, menor pero hay que decirla: cargar_nombres() pasa todo a minusculas, asi
que el PATRON DE MAYUSCULAS («HOW_TO_DECRYPT» contra «How_To_Decrypt») NO se puede probar
aunque es una de las abstracciones naturales. Y el «[]» que marca el ID de la victima lo puso
la auditoria a mano, no el ransomware: «tiene ID» se mide sobre una cadena ya normalizada por
una persona.

=============================================================================================
PREREGISTRO -- escrito y COMMITEADO ANTES de correr (2026-09-28).

H1. PUERTA DE ENTRADA. La cascada SIN la capa nueva reproduce la cifra de cabecera P2bal:
    macro-F1 0,7417 y exactitud 0,8123 (tolerancia 0,01, fuente 4_resultados/_log_p2bal_149.txt).
    Si no, sys.exit y NO se reporta nada.
H1b. CONTROL INTERNO. El diccionario y la regla locales, con CERO abstracciones, tienen que dar
    predicciones IDENTICAS a protocolo_logo.dicc_privados / protocolo_logo.regla. Si no, el
    codigo nuevo cambio la cascada y todo lo que siga es incomparable. Si no, sys.exit.
H2. TECHO. Las notas con nombre usable son exactamente 64 de 149 = 0,4295. Se verifica en
    codigo. La cobertura de CUALQUIER variante nunca puede superar la cobertura base + 0,4295.
H3. COBERTURA AGREGADA. Las cuatro abstracciones candidatas (ESQ, ESQC, FIRMA, DOM) agregan
    cobertura ESTRICTAMENTE POSITIVA: disparan en notas donde ni los IOCs ni el nombre exacto
    disparaban. Prediccion: entre 0,010 y 0,100 del corpus (entre ~1,5 y ~15 notas de 149).
    Si la cobertura agregada es 0, la forma no aporta NADA que el nombre exacto no diera ya, y
    el experimento termina ahi.
H4. LA PREGUNTA PRINCIPAL. Al menos una de las cuatro da Delta macro-F1 GLOBAL > 0 con IC 95 %
    de t-Student que EXCLUYE el cero, sobre 50 semillas pareadas.
H5. LA GANADORA SERA **DOM**, no las geometricas. Razon: en el corpus las variantes que hay que
    unificar se diferencian por PUNTUACION con el mismo contenido lexico (CERBER
    «# decrypt my files #» contra «# decrypt_my_files #»), y DOM las une mientras ESQ las separa.
    Prediccion numerica: Delta macro-F1 global de DOM entre +0,005 y +0,040.
    Si gana una geometrica, la senal del nombre en notas es de molde y no de vocabulario, que es
    lo contrario de lo que predice el corpus y hay que decirlo.
H6. CONTROL EXT. La extension sola da cobertura agregada <= 0,02 y |Delta macro-F1| <= 0,005,
    porque el filtro de genericos la borra salvo «.htm» (solo TESLACRYPT, 2 notas). Si EXT
    mueve mas que eso, el filtro de genericos no funciona y se invalida el resto.
H7. IDENTIDAD ARITMETICA (control, no prediccion). En EXACTITUD, la capa solo puede cambiar
    notas con nombre, asi que tiene que cumplirse EXACTAMENTE
        Delta exactitud restringida = Delta exactitud global * 149/64.
    Si no se cumple, la capa esta tocando notas sin nombre: es un BUG y aborta la lectura.
H8. COLAPSAR PIERDE. Delta(ESQC) <= Delta(ESQ). Razon: sobre los 39 nombres distintos del
    corpus el esqueleto crudo deja 26 valores privados de una sola familia y el colapsado solo
    19; el colapso destruye mas poder discriminante del que gana en generalizacion.
H9. LA COMBINACION NO SUMA. Delta(TODAS) <= max(Delta individual). Razon: las abstracciones
    estan anidadas (ESQC es mas gruesa que ESQ, EXT esta contenida en todas) y agregar claves
    bajo unanimidad solo puede ROMPER acuerdos, nunca crearlos.
H10. LECTURA FIJADA DE ANTEMANO, para no racionalizar despues:
    - Si H4 se cumple: el hallazgo del Exp. 2d TRANSFIERE al frente de notas -- la forma del
      nombre complementa al texto como complementaba a los bytes -- pero con techo 42,95 %.
    - Si H4 falla Y el acierto en lo agregado es alto (>= 0,80): la capa ACIERTA pero es
      demasiado chica para mover el agregado. Mismo veredicto que el POSTHOC de LOGO: regla de
      alta precision y baja cobertura. NO se adopta, y la razon es el techo del corpus.
    - Si H4 falla Y el acierto en lo agregado es bajo (< 0,80): la forma del nombre NO es firma
      de familia en notas, y la transferencia del Exp. 2d FALLA. La explicacion coherente con
      los dos frentes es que en NapierOne cada familia es UNA campana -- la forma del nombre ES
      la campana, que es la limitacion ya declarada del 2d -- mientras que en el corpus de notas
      cada familia trae notas de varias campanas, que renombran distinto.

NOTA DE HONESTIDAD: las cinco abstracciones se disenaron MIRANDO los 39 nombres distintos del
corpus (no hay forma de disenarlas a ciegas). Por lo tanto este experimento NO valida el
criterio de abstraccion: mide si, DADAS estas abstracciones, la forma aporta. Se declara asi.
Lo que NO se miro antes de commitear este archivo es cualquier resultado de la cascada.
=============================================================================================
=============================================================================================
AMPLIACION DEL 2026-09-28 -- POSTERIOR A LA CORRIDA ORIGINAL. NO REESCRIBE NADA DE ARRIBA.

Todo lo que esta ENCIMA de esta linea quedo commiteado ANTES de correr (commit 5b23894) y se
deja tal cual, incluidas las predicciones que fallaron. Este bloque SE AGREGA.

MOTIVO: la corrida original ya habia terminado cuando llego el aviso de que la cifra que este
script cita como justificacion -- los 0,9998 de «bytes + forma del nombre» del frente de
archivos -- esta COMPROMETIDA POR UNA FUGA. Fuente: 2_codigo/exp2g_nombre_robusto.py
(commit 4477dbb, 2026-09-28).

QUE PASO EN EL FRENTE DE ARCHIVOS. El Exp. 2f midio el sistema completo en 0,9998, pero su
validacion por tipos de documento colapso en un pliegue: con los jpg fuera del entrenamiento,
sumar la forma del nombre bajo el macro-F1 de 0,8052 a 0,2167. La causa se verifico mirando
los archivos:
    0001-doc.doc.avos2      0001-pdf.pdf.avos2      0001-jpg-fromweb.jpg.avos2
La BASE del nombre la puso NapierOne al armar su corpus, y en los jpg lleva un «-fromweb» que
en los documentos no esta. La funcion de forma miraba el nombre ENTERO, asi que aprendio
tambien como nombro NAPIERONE sus archivos, no solo como renombra el ransomware. El arreglo
del Exp. 2g fue calcular los rasgos SOLO sobre la extension final, que es lo unico que agrega
el ransomware.

CONSECUENCIA PARA LA JUSTIFICACION DE ESTE SCRIPT. El «0,9998» citado arriba NO se puede usar
como evidencia de que la forma del nombre funciona: incluye un componente que aprendia el
esquema de nombrado del curador. La evidencia de transferencia esta siendo RE-MEDIDA por el
Exp. 2g y, mientras no cierre, este experimento NO se apoya en esa cifra. Lo que queda en pie
del 2d es mas debil y mas honesto: la forma del nombre SOLA daba 0,5771, o sea que nunca
resolvio la tarea por si misma. La pregunta de este script sigue siendo valida -- ¿aporta la
forma del nombre en notas? -- pero ya no viene respaldada por un 0,9998.

EL MISMO RIESGO EN ESTE SCRIPT, Y DONDE ESTA EXACTAMENTE.
Que un nombre sea AUDITADO garantiza que es el nombre genuino que puso el ransomware; NO
garantiza que todas las partes de la cadena sean informativas de la familia. Aca la parte
contaminada esta identificada y VERIFICADA (no razonada): el ID de la victima lo enmascaro a
mano la auditoria, y lo hizo con DOS NOTACIONES DISTINTAS:
    «[]»             en 15 de los 16 nombres que llevan ID
    «[victim's_id]»  en 1  (readme.[victim's_id].txt, DARKSIDE)
Esas dos notaciones tienen 2 y 13 caracteres. Por lo tanto el LARGO de la cadena y la CANTIDAD
DE TOKENS no son propiedades del ransomware: dependen de como escribio la mascara una persona.
Y el efecto es concreto: el unico nombre de DARKSIDE cae en el tramo «largo» por la notacion, y
caeria en «medio» si la mascara fuera «[]» como en los otros 15. En FIRMA ese nombre produce la
clave privada «.txt|none|1|no|SI|largo» -> DARKSIDE, cuyo campo «largo» es un artefacto de la
auditoria. Es la misma forma de fuga que el «-fromweb» de NapierOne, a menor escala.

LOS DOS GRUPOS, QUE SE REPORTAN POR SEPARADO (pedido del 2026-09-28):

  GRUPO A -- SOLIDAS. Dependen SOLO de lo que pone el ransomware, y son inmunes a la notacion
  de la mascara porque borran el ID antes de medir o no miran largos:
      EXT   la extension final (control)
      DOM   palabras del dominio + extension (borra ID, separadores y adornos)
      ROB   extension + hay ID + palabras del dominio  [NUEVA]
      FROB  extension + separador dominante + hay digitos + hay ID  [NUEVA]
            (es FIRMA SIN los dos campos contaminados: largo y cantidad de tokens)

  GRUPO B -- SOSPECHOSAS. Miran el nombre ENTERO: largo literal, cantidad de tokens, tipografia.
      ESQ · ESQC · FIRMA

  Regla de lectura fijada ahora: si el aporte viene del GRUPO B, es SOSPECHOSO y se declara
  como tal; si viene del GRUPO A, es solido. Se mide ademas cada grupo combinado.

LIMITACION ADICIONAL QUE SE DECLARA (pedido del 2026-09-28): el corpus de notas viene de
repositorios publicos que RENOMBRAN AL CATALOGAR. Por eso solo 64 de 149 notas conservan el
nombre genuino y las otras 85 llegan como None. La via del nombre tiene entonces un techo que
NO es del metodo sino de como se distribuyen las notas: aunque la abstraccion fuera perfecta,
no puede pasar del 42,95 % del corpus, y ese numero solo sube consiguiendo notas de fuentes que
preserven el nombre original.

HIPOTESIS DE LA AMPLIACION -- CON SU GRADO DE CEGUERA DECLARADO:
  A1. [CIEGA] ROB y FROB (grupo A, nunca medidas) dan Delta macro-F1 global <= 0, o con IC que
      incluye el cero. Razon: DOM, que es del mismo grupo y ya estaba medida, dio -0,0067 con
      0/50 semillas positivas.
  A2. [NO CIEGA -- es una OBSERVACION de la corrida original, no una prediccion] la unica
      abstraccion con Delta macro-F1 global positivo y IC que excluye el cero es ESQ, que
      pertenece al GRUPO B. O sea: el unico aporte nominal del experimento viene del grupo
      sospechoso. Se verifica en codigo y se reporta como tal.
  A3. [CIEGA] GRUPO_A combinado da Delta macro-F1 global <= 0.
  A4. [CIEGA, verificable sin correr] hay al menos 2 notaciones distintas de la mascara del ID,
      lo que demuestra que largo y cantidad de tokens dependen del auditor.
  A5. [CIEGA] FROB (FIRMA sin largo ni tokens) no da MENOS que FIRMA. Si FIRMA aportara algo
      real de estructura, sacarle los dos campos contaminados no deberia destruirlo. Si FROB
      cae respecto de FIRMA, lo que FIRMA aportaba estaba en los campos contaminados.

NOTA DE HONESTIDAD DE LA AMPLIACION: este bloque se escribe DESPUES de haber visto los
resultados de la corrida original. Las hipotesis marcadas [NO CIEGA] no son predicciones y se
reportan como observaciones. Las marcadas [CIEGA] se refieren a variantes que no se habian
medido todavia cuando se escribio este bloque.
=============================================================================================

Uso:  python capa_forma_nombre.py [--n-semillas 50] [--salida CARPETA]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import obtener_modelos, vectorizador
from protocolo_logo import dicc_privados, regla
from protocolo_p2bal import split_p2bal
from revision_logo import cargar_todo

RAIZ = _AQUI.parent
OUT_DEF = RAIZ / "4_resultados" / "resultados_capa_forma_nombre_149"
CANON_F1, CANON_ACC, TOL = 0.7417, 0.8123, 0.01     # fuente: 4_resultados/_log_p2bal_149.txt
N_CON_NOMBRE_ESPERADO = 64                          # H2: techo declarado
N_NOTAS_ESPERADO = 149


# ============================================================
# LAS ABSTRACCIONES DE LA FORMA DEL NOMBRE
# ============================================================
RE_ID = re.compile(r"\[[^\]]*\]")      # el ID de la victima, ya enmascarado por la auditoria
RE_LET = re.compile(r"[a-z]+")
RE_DIG = re.compile(r"\d+")

# Vocabulario fijo del dominio, cerrado ANTES de correr. Se marca por subcadena sobre el nombre
# sin separadores, asi que las palabras anidadas ("read" dentro de "readme") disparan las dos:
# es deterministico y se aplica igual a todas las notas.
VOCAB_DOMINIO = ("decrypt", "encrypt", "crypt", "restore", "recover", "readme", "read",
                 "unlock", "files", "file", "help", "how", "back", "info", "return", "faq",
                 "security", "event", "this", "me", "my", "please", "important", "note",
                 "lock", "data", "instruction", "message", "warning", "attention", "key")


def _partir(nombre):
    """Parte en (base, extension-con-punto) por el ULTIMO punto. Sin punto -> extension vacia."""
    if "." in nombre:
        base, ext = nombre.rsplit(".", 1)
        return base, "." + ext
    return nombre, ""


def abs_esqueleto(nombre):
    """ESQ: letras -> W, digitos -> D, ID -> [], el resto literal. Conserva la extension."""
    base, ext = _partir(nombre)
    b = RE_ID.sub("[]", base)
    b = RE_DIG.sub("D", b)
    b = RE_LET.sub("W", b)
    return b + ext


def abs_esqueleto_colapsado(nombre):
    """ESQC: el esqueleto con las cadenas «W<sep>W<sep>...» colapsadas en «W+»."""
    e = abs_esqueleto(nombre)
    for sep in ("_", "-", " "):
        e = re.sub(r"W(?:%sW)+" % re.escape(sep), "W+", e)
    return e


def abs_firma(nombre):
    """FIRMA: extension · separador dominante · n tokens · digitos · ID · largo, todo en tramos."""
    base, ext = _partir(nombre)
    tiene_id = "SI" if RE_ID.search(base) else "no"
    b = RE_ID.sub("", base)
    dig = "SI" if RE_DIG.search(b) else "no"
    cuenta = {s: b.count(s) for s in ("_", "-", " ")}
    sep = max(cuenta, key=lambda s: cuenta[s]) if max(cuenta.values()) > 0 else "none"
    toks = [t for t in re.split(r"[^a-z0-9]+", b) if t]
    n_tok = str(len(toks)) if len(toks) <= 3 else "4+"
    largo = len(base)
    tramo = ("corto" if largo <= 8 else "medio" if largo <= 16
             else "largo" if largo <= 24 else "muylargo")
    return "|".join([ext, sep, n_tok, dig, tiene_id, tramo])


def abs_dominio(nombre):
    """DOM: conjunto ordenado de palabras del dominio presentes + extension. Sin andamiaje."""
    base, ext = _partir(nombre)
    letras = re.sub(r"[^a-z0-9]", "", RE_ID.sub("", base))
    pal = sorted({w for w in VOCAB_DOMINIO if w in letras})
    return ("+".join(pal) if pal else "SIN") + ext


def abs_extension(nombre):
    """EXT: solo la extension. Es el CONTROL del filtro de genericos, no una candidata."""
    return _partir(nombre)[1] or "SIN_EXT"


# ---- AMPLIACION 2026-09-28: abstracciones del GRUPO A (solo lo que pone el ransomware) ----
def abs_robusta(nombre):
    """ROB: extension + hay ID de victima + palabras del dominio. NO mira largos ni tokens,
    asi que no depende de con que notacion la auditoria enmascaro el ID."""
    base, ext = _partir(nombre)
    tiene_id = "SI" if RE_ID.search(base) else "no"
    letras = re.sub(r"[^a-z0-9]", "", RE_ID.sub("", base))
    pal = sorted({w for w in VOCAB_DOMINIO if w in letras})
    return "|".join([ext, tiene_id, "+".join(pal) if pal else "SIN"])


def abs_firma_robusta(nombre):
    """FROB: la FIRMA sin los dos campos contaminados por la auditoria (largo y n de tokens).
    Queda extension + separador dominante + hay digitos + hay ID. Ablacion que localiza de
    donde salia lo que FIRMA aportaba."""
    base, ext = _partir(nombre)
    tiene_id = "SI" if RE_ID.search(base) else "no"
    b = RE_ID.sub("", base)
    dig = "SI" if RE_DIG.search(b) else "no"
    cuenta = {s: b.count(s) for s in ("_", "-", " ")}
    sep = max(cuenta, key=lambda s: cuenta[s]) if max(cuenta.values()) > 0 else "none"
    return "|".join([ext, sep, dig, tiene_id])


ABSTRACCIONES = {
    "ESQ": abs_esqueleto,
    "ESQC": abs_esqueleto_colapsado,
    "FIRMA": abs_firma,
    "DOM": abs_dominio,
    "EXT": abs_extension,
    "ROB": abs_robusta,
    "FROB": abs_firma_robusta,
}
CANDIDATAS = ["ESQ", "ESQC", "FIRMA", "DOM"]        # EXT queda fuera: es control
# Los dos grupos de la ampliacion. A = solo lo que pone el ransomware. B = mira el nombre entero.
GRUPO_A = ["EXT", "DOM", "ROB", "FROB"]
GRUPO_B = ["ESQ", "ESQC", "FIRMA"]
# Las variantes fijadas antes de correr. "base" = la cascada de siempre, sin capa de forma.
# TODAS es la combinacion original (las 4 candidatas del preregistro, sin tocar).
VARIANTES = (["base"] + list(ABSTRACCIONES) + ["TODAS", "GRUPO_A", "GRUPO_B"])
COMBOS = {"TODAS": CANDIDATAS, "GRUPO_A": GRUPO_A, "GRUPO_B": GRUPO_B}


# ============================================================
# DICCIONARIO Y REGLA con las claves de forma AGREGADAS
# Replican exacto protocolo_logo.dicc_privados / regla y le suman las claves de forma.
# Con abs_usadas = [] tienen que dar lo mismo que el original (control H1b).
# ============================================================
def claves_de(i, iocs, nombres_nota, abs_usadas):
    """Todas las claves de la nota i: los IOCs, el nombre exacto y una por abstraccion."""
    cl = set(iocs[i])
    nm = nombres_nota[i]
    if nm:
        cl.add(("[NOMBRE]", nm))
        for tag in abs_usadas:
            cl.add(("[FORMA_%s]" % tag, ABSTRACCIONES[tag](nm)))
    return cl


def dicc_con_forma(tr, iocs, nombres_nota, y, abs_usadas):
    """Mismo filtro de genericos que la cascada: toda clave que en TRAIN apunta a mas de una
    familia se borra. El diccionario se arma SOLO con el pliegue de entrenamiento."""
    d = defaultdict(set)
    for i in tr:
        for c in claves_de(i, iocs, nombres_nota, abs_usadas):
            d[c].add(y[i])
    for k in [k for k, v in d.items() if len(v) > 1]:
        del d[k]
    return d


def regla_con_forma(i, d, iocs, nombres_nota, abs_usadas):
    """Misma regla de unanimidad: responde solo si TODAS las claves que disparan coinciden."""
    fams = set()
    for c in claves_de(i, iocs, nombres_nota, abs_usadas):
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
    print("  CAPA DE FORMA DEL NOMBRE -- transferencia del Exp. 2d al frente de notas")
    print("=" * 90)
    textos, textos_arr, y, archivos, grupos, iocs, nombres_nota = cargar_todo()
    familias = np.unique(y)
    n = len(y)
    con_nombre = np.array([bool(nm) for nm in nombres_nota])
    n_con = int(con_nombre.sum())
    fam_con_nombre = np.unique(y[con_nombre])

    print(f"Notas: {n} | Familias: {len(familias)} | Plantillas: {len(set(grupos))} | Semillas: {S}")
    print(f"Protocolo: P2bal (split_p2bal, rng = default_rng(20000 + s))")

    # ---------------- H2: el techo, antes de cualquier otra cifra ----------------
    techo = n_con / n
    print("\n" + "-" * 90)
    print("  H2 -- TECHO ESTRUCTURAL (se declara antes de todo lo demas)")
    print("-" * 90)
    print(f"  Notas con nombre genuino auditado: {n_con}/{n} = {techo:.4f}  ({100*techo:.2f} %)")
    print(f"  Nombres exactos distintos: {len({nm for nm in nombres_nota if nm})}")
    print(f"  Familias con al menos una nota con nombre: {len(fam_con_nombre)}/{len(familias)}")
    print(f"  >>> NINGUNA capa basada en el nombre puede cubrir mas del {100*techo:.2f} % del corpus.")
    print(f"  >>> Las otras {n - n_con} notas fueron renombradas por quien las recolecto:")
    print( "  >>> usar esos nombres seria CIRCULAR y por eso llegan como None.")
    ok_h2 = (n == N_NOTAS_ESPERADO and n_con == N_CON_NOMBRE_ESPERADO)
    print(f"  H2 {'CUMPLE' if ok_h2 else 'FALLA'}: esperado {N_CON_NOMBRE_ESPERADO}/{N_NOTAS_ESPERADO}")
    print( "  >>> El corpus de notas viene de repositorios publicos que RENOMBRAN AL CATALOGAR.")
    print( "  >>> El techo no es del metodo: es de como se distribuyen las notas.")

    # ---------------- A4: sonda de fuga, al estilo del Exp. 2g ----------------
    # El «-fromweb» de NapierOne enseno que hay que mirar QUE PARTE del nombre puso el
    # ransomware y que parte puso el curador. Aca la parte del curador es la mascara del ID.
    print("\n" + "-" * 90)
    print("  A4 -- SONDA DE FUGA: ¿que parte del nombre la puso una persona y no el ransomware?")
    print("-" * 90)
    nom_unicos = sorted({nm for nm in nombres_nota if nm})
    masc = defaultdict(list)
    for nm in nom_unicos:
        for m in RE_ID.findall(nm):
            masc[m].append(nm)
    print(f"  Notaciones distintas de la mascara del ID de la victima: {len(masc)}")
    for m, lst in sorted(masc.items(), key=lambda kv: -len(kv[1])):
        print(f"    {m!r:<18} en {len(lst)} nombres (largo {len(m)} caracteres)")
    cambia = []
    for nm in nom_unicos:
        if RE_ID.search(nm):
            b = _partir(nm)[0]
            if abs_firma(nm) != abs_firma(nm.replace(RE_ID.search(nm).group(0), "[]")):
                cambia.append((nm, abs_firma(nm),
                               abs_firma(nm.replace(RE_ID.search(nm).group(0), "[]"))))
    ok_a4 = len(masc) >= 2
    print(f"  A4 {'CUMPLE' if ok_a4 else 'FALLA'}: con {len(masc)} notaciones, el LARGO y la "
          "CANTIDAD DE TOKENS")
    print( "      dependen del auditor, no del ransomware. Nombres cuya FIRMA cambia si se")
    print(f"      uniformara la mascara a «[]»: {len(cambia)}")
    for nm, f_real, f_unif in cambia:
        print(f"        {nm}\n          firma real      {f_real}\n          firma uniformada {f_unif}")

    # ---------------- inventario de valores por abstraccion (descripcion del corpus) -------
    inv = []
    pares = sorted({(f, nm) for f, nm in zip(y, nombres_nota) if nm})
    for tag, fn in ABSTRACCIONES.items():
        mapa = defaultdict(set)
        for f, nm in pares:
            mapa[fn(nm)].add(f)
        priv = sum(1 for v in mapa.values() if len(v) == 1)
        inv.append(dict(abstraccion=tag, valores_distintos=len(mapa), valores_privados=priv,
                        valores_genericos=len(mapa) - priv))
        for val in sorted(mapa):
            inv.append(dict(abstraccion=tag, valor=val, familias="+".join(sorted(mapa[val])),
                            n_familias=len(mapa[val])))
    pd.DataFrame(inv).to_csv(OUT / "forma_valores.csv", index=False, encoding="utf-8-sig")
    print("\n  Inventario sobre los 39 nombres distintos (corpus entero, solo descriptivo):")
    print("    abstraccion  valores  privados(1 familia)  genericos(>1, el filtro los borra)")
    for r in inv:
        if "valores_distintos" in r:
            print(f"    {r['abstraccion']:<12} {r['valores_distintos']:>7} {r['valores_privados']:>20}"
                  f" {r['valores_genericos']:>34}")

    # ---------------- evaluacion ----------------
    # El clasificador de TEXTO no depende de la capa de forma: se entrena UNA vez por
    # (semilla, pliegue) y se reusa en todas las variantes. Asi la comparacion es pareada
    # exacta: entre variantes cambia SOLO la capa de reglas.
    pred = {v: np.empty((S, n), dtype=object) for v in VARIANTES}
    aplica = {v: np.zeros((S, n), dtype=bool) for v in VARIANTES}
    control_h1b = True

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

            # control H1b: el original, tal cual esta en protocolo_logo
            d_orig = dicc_privados(tr, iocs, nombres_nota, y)

            for v in VARIANTES:
                usadas = [] if v == "base" else COMBOS.get(v, [v])
                d = dicc_con_forma(tr, iocs, nombres_nota, y, usadas)
                if v == "base" and d != d_orig:
                    control_h1b = False
                for k, i in enumerate(te):
                    r = regla_con_forma(i, d, iocs, nombres_nota, usadas)
                    if v == "base" and r != regla(i, d_orig, iocs, nombres_nota):
                        control_h1b = False
                    pred[v][s, i] = pt[k] if r is None else r
                    aplica[v][s, i] = r is not None
        if (s + 1) % 10 == 0:
            print(f"  {s+1}/{S} semillas")

    # ---------------- H1 / H1b: puerta de entrada ----------------
    f1 = {v: np.array([f1_score(y, pred[v][s], average="macro", labels=familias, zero_division=0)
                       for s in range(S)]) for v in VARIANTES}
    ac = {v: np.array([accuracy_score(y, pred[v][s]) for s in range(S)]) for v in VARIANTES}

    print("\n" + "-" * 90)
    print("  H1 -- PUERTA DE ENTRADA (la cascada SIN la capa nueva)")
    print("-" * 90)
    print(f"  base: macro-F1 {f1['base'].mean():.4f} vs {CANON_F1} | "
          f"exactitud {ac['base'].mean():.4f} vs {CANON_ACC}  (tolerancia {TOL})")
    ok_h1 = abs(f1["base"].mean() - CANON_F1) <= TOL and abs(ac["base"].mean() - CANON_ACC) <= TOL
    print(f"  H1b control interno (diccionario y regla locales == protocolo_logo): "
          f"{'IDENTICOS' if control_h1b else 'DIFIEREN'}")
    if (not ok_h1 or not control_h1b) and not args.sin_puerta:
        sys.exit("ABORTADO (H1/H1b): la cascada base no reproduce la cabecera o el codigo "
                 "local no replica protocolo_logo. No se reporta nada.")
    print("  OK\n" if ok_h1 and control_h1b else "  FUERA DE TOLERANCIA (--sin-puerta)\n")

    # ---------------- las TRES COLUMNAS por variante ----------------
    idx_con = np.where(con_nombre)[0]
    filas = []
    for v in VARIANTES:
        ap_v, ap_b = aplica[v], aplica["base"]
        cob = float(ap_v.mean())
        acd = float((pred[v][ap_v] == np.tile(y, (S, 1))[ap_v]).mean()) if ap_v.any() else np.nan
        # lo que la capa AGREGA: donde la base no disparaba y la variante si
        m_add = ap_v & ~ap_b
        cob_add = float(m_add.mean())
        ac_add = float((pred[v][m_add] == np.tile(y, (S, 1))[m_add]).mean()) if m_add.any() else np.nan
        # lo que la capa ROMPE: donde la base disparaba y la variante ya no (unanimidad rota)
        m_rot = ap_b & ~ap_v
        cob_rot = float(m_rot.mean())
        ac_rot = float((pred["base"][m_rot] == np.tile(y, (S, 1))[m_rot]).mean()) if m_rot.any() else np.nan
        # lo que la capa CAMBIA de respuesta (disparaban las dos, distinto resultado)
        m_cam = ap_v & ap_b & (pred[v] != pred["base"])
        grupo = ("-" if v == "base" else "A solida" if v in GRUPO_A
                 else "B sospechosa" if v in GRUPO_B else "combo")
        filas.append(dict(
            variante=v,
            grupo=grupo,
            cobertura=round(cob, 4),
            acierto_donde_aplica=round(acd, 4) if acd == acd else np.nan,
            macro_f1=round(float(f1[v].mean()), 4),
            exactitud=round(float(ac[v].mean()), 4),
            exact_balanceada=round(float(np.mean([balanced_accuracy_score(y, pred[v][s])
                                                  for s in range(S)])), 4),
            mcc=round(float(np.mean([matthews_corrcoef(y, pred[v][s]) for s in range(S)])), 4),
            cobertura_agregada=round(cob_add, 4),
            acierto_en_lo_agregado=round(ac_add, 4) if ac_add == ac_add else np.nan,
            cobertura_rota=round(cob_rot, 4),
            acierto_base_en_lo_roto=round(ac_rot, 4) if ac_rot == ac_rot else np.nan,
            frac_respuesta_cambiada=round(float(m_cam.mean()), 4),
        ))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "forma_resumen.csv", index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("  LAS TRES COLUMNAS (149 notas, P2bal, 50 semillas). Techo de cobertura del nombre: "
          f"{techo:.4f}")
    print("=" * 90)
    print(df[["variante", "grupo", "cobertura", "acierto_donde_aplica", "macro_f1", "exactitud"]]
          .to_string(index=False))
    print("\n  Desglose de lo que la capa agrega y de lo que rompe:")
    print(df[["variante", "cobertura_agregada", "acierto_en_lo_agregado", "cobertura_rota",
              "acierto_base_en_lo_roto", "frac_respuesta_cambiada"]].to_string(index=False))

    # ---------------- Deltas pareados por semilla: GLOBAL y RESTRINGIDO ----------------
    yc = y[idx_con]
    f1_r = {v: np.array([f1_score(yc, pred[v][s, idx_con], average="macro",
                                  labels=fam_con_nombre, zero_division=0) for s in range(S)])
            for v in VARIANTES}
    ac_r = {v: np.array([accuracy_score(yc, pred[v][s, idx_con]) for s in range(S)])
            for v in VARIANTES}

    dfilas = []
    for v in VARIANTES:
        if v == "base":
            continue
        for etiqueta, a, b in (("macro_f1_global", f1[v], f1["base"]),
                               ("exactitud_global", ac[v], ac["base"]),
                               ("macro_f1_restringido", f1_r[v], f1_r["base"]),
                               ("exactitud_restringida", ac_r[v], ac_r["base"])):
            d = a - b
            m, lo, hi = ic(d)
            dfilas.append(dict(variante=v, metrica=etiqueta, delta=round(m, 4),
                               ic_bajo=round(lo, 4), ic_alto=round(hi, 4),
                               semillas_positivas=int((d > 0).sum()), n_semillas=S,
                               ic_excluye_cero="SI" if (lo > 0 or hi < 0) else "no"))
    dd = pd.DataFrame(dfilas)
    dd.to_csv(OUT / "forma_deltas.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("  DELTA PAREADO POR SEMILLA contra la cascada actual (IC 95 % t-Student, n = 50)")
    print("  GLOBAL = las 149 notas · RESTRINGIDO = solo las 64 con nombre genuino")
    print("=" * 90)
    for v in VARIANTES:
        if v == "base":
            continue
        print(f"\n  --- {v} ---")
        for _, r in dd[dd.variante == v].iterrows():
            print(f"    {r['metrica']:<24} D {r['delta']:+.4f} "
                  f"[{r['ic_bajo']:+.4f}; {r['ic_alto']:+.4f}]  "
                  f"{r['semillas_positivas']}/{S} semillas  IC excluye 0: {r['ic_excluye_cero']}")

    # ---------------- SELECCION post-hoc (declarada) ----------------
    d_f1_glob = {v: float((f1[v] - f1["base"]).mean()) for v in CANDIDATAS}
    sel = [v for v in CANDIDATAS if d_f1_glob[v] > 0]
    print("\n" + "-" * 90)
    print("  SELECCION POST-HOC (se declara como post-hoc: se eligio MIRANDO los resultados)")
    print("-" * 90)
    print(f"  Abstracciones con Delta macro-F1 global medio > 0: {sel if sel else 'NINGUNA'}")
    fila_sel = None
    if sel and set(sel) != set(CANDIDATAS):
        ps = np.empty((S, n), dtype=object)
        aps = np.zeros((S, n), dtype=bool)
        for s in range(S):
            rng = np.random.default_rng(20_000 + s)
            for tr, te in split_p2bal(y, grupos, familias, rng):
                vec = vectorizador("combinado")
                Xtr = vec.fit_transform(textos_arr[tr])
                clf = obtener_modelos(s)["LinearSVC"]
                clf.fit(Xtr, y[tr])
                pt = clf.predict(vec.transform(textos_arr[te]))
                d = dicc_con_forma(tr, iocs, nombres_nota, y, sel)
                for k, i in enumerate(te):
                    r = regla_con_forma(i, d, iocs, nombres_nota, sel)
                    ps[s, i] = pt[k] if r is None else r
                    aps[s, i] = r is not None
        f1s = np.array([f1_score(y, ps[s], average="macro", labels=familias, zero_division=0)
                        for s in range(S)])
        acs = np.array([accuracy_score(y, ps[s]) for s in range(S)])
        m, lo, hi = ic(f1s - f1["base"])
        fila_sel = dict(seleccion="+".join(sel), cobertura=round(float(aps.mean()), 4),
                        macro_f1=round(float(f1s.mean()), 4), exactitud=round(float(acs.mean()), 4),
                        delta_macro_f1=round(m, 4), ic_bajo=round(lo, 4), ic_alto=round(hi, 4),
                        semillas_positivas=int(((f1s - f1["base"]) > 0).sum()))
        pd.DataFrame([fila_sel]).to_csv(OUT / "forma_seleccion_posthoc.csv", index=False,
                                        encoding="utf-8-sig")
        print(f"  SELECCION({'+'.join(sel)}): macro-F1 {f1s.mean():.4f} | "
              f"D {m:+.4f} [{lo:+.4f}; {hi:+.4f}] {fila_sel['semillas_positivas']}/{S}")
    elif sel:
        print("  Coincide con TODAS: no se re-corre (ya esta arriba).")

    # ---------------- por familia, mejor candidata ----------------
    mejor = max(CANDIDATAS, key=lambda v: d_f1_glob[v])
    ff = []
    for f in familias:
        idx = np.where(y == f)[0]
        n_nom = int(con_nombre[idx].sum())
        ff.append(dict(familia=f, n_notas=len(idx), n_con_nombre=n_nom,
                       base=round(float(np.mean([(pred["base"][s, idx] == f).mean()
                                                 for s in range(S)])), 4),
                       mejor=round(float(np.mean([(pred[mejor][s, idx] == f).mean()
                                                  for s in range(S)])), 4)))
    dff = pd.DataFrame(ff)
    dff["delta"] = (dff.mejor - dff.base).round(4)
    dff = dff.sort_values("delta", ascending=False)
    dff.to_csv(OUT / "forma_por_familia.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== POR FAMILIA: base contra la mejor candidata ({mejor}) ===")
    print(dff[dff.n_con_nombre > 0].to_string(index=False))

    # ---------------- VEREDICTO ----------------
    def getd(v, met):
        r = dd[(dd.variante == v) & (dd.metrica == met)].iloc[0]
        return float(r["delta"]), float(r["ic_bajo"]), float(r["ic_alto"]), int(r["semillas_positivas"])

    h3 = all(df[df.variante == v].cobertura_agregada.iloc[0] > 0 for v in CANDIDATAS)
    h3_rango = all(0.010 <= df[df.variante == v].cobertura_agregada.iloc[0] <= 0.100
                   for v in CANDIDATAS)
    h4_v = [v for v in CANDIDATAS if getd(v, "macro_f1_global")[1] > 0]
    dom_d = getd("DOM", "macro_f1_global")[0]
    h5 = (mejor == "DOM") and (0.005 <= dom_d <= 0.040)
    ext_cob = df[df.variante == "EXT"].cobertura_agregada.iloc[0]
    h6 = ext_cob <= 0.02 and abs(getd("EXT", "macro_f1_global")[0]) <= 0.005
    # H7: identidad aritmetica exactitud restringida = global * 149/64
    fac = n / n_con
    h7_err = max(abs(getd(v, "exactitud_restringida")[0] - getd(v, "exactitud_global")[0] * fac)
                 for v in VARIANTES if v != "base")
    h7 = h7_err <= 5e-4
    h8 = d_f1_glob["ESQC"] <= d_f1_glob["ESQ"]
    h9 = getd("TODAS", "macro_f1_global")[0] <= max(d_f1_glob[v] for v in CANDIDATAS) + 1e-9
    mejor_add = df[df.variante == mejor].acierto_en_lo_agregado.iloc[0]

    print("\n" + "=" * 90)
    print("  VEREDICTO DEL PREREGISTRO (se reporta igual lo que falle)")
    print("=" * 90)
    chk = [
        ("H1 puerta de entrada", ok_h1, f"{f1['base'].mean():.4f} / {ac['base'].mean():.4f}"),
        ("H1b control interno == protocolo_logo", control_h1b,
         "identicos" if control_h1b else "DIFIEREN"),
        ("H2 techo 64/149 = 0,4295", ok_h2, f"{n_con}/{n} = {techo:.4f}"),
        ("H3 las 4 candidatas agregan cobertura > 0", h3,
         " · ".join(f"{v} {df[df.variante==v].cobertura_agregada.iloc[0]:.4f}" for v in CANDIDATAS)),
        ("H3b cobertura agregada en [0,010; 0,100]", h3_rango, ""),
        ("H4 alguna da Delta macro-F1 global > 0 con IC que excluye 0", bool(h4_v),
         str(h4_v) if h4_v else "ninguna"),
        ("H5 la ganadora es DOM y su Delta cae en [+0,005; +0,040]", h5,
         f"mejor = {mejor} · DOM {dom_d:+.4f}"),
        ("H6 control EXT: cobertura <= 0,02 y |Delta| <= 0,005", h6,
         f"cob_add {ext_cob:.4f} · D {getd('EXT','macro_f1_global')[0]:+.4f}"),
        ("H7 identidad exactitud restringida = global * 149/64", h7, f"error max {h7_err:.6f}"),
        ("H8 Delta(ESQC) <= Delta(ESQ)", h8,
         f"ESQC {d_f1_glob['ESQC']:+.4f} vs ESQ {d_f1_glob['ESQ']:+.4f}"),
        ("H9 Delta(TODAS) <= max individual", h9,
         f"TODAS {getd('TODAS','macro_f1_global')[0]:+.4f} vs max "
         f"{max(d_f1_glob.values()):+.4f}"),
    ]
    for nombre, cumple, det in chk:
        print(f"  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<56} {det}")

    print("\n  LECTURA SEGUN H10 (fijada antes de correr):")
    if h4_v:
        print(f"    TRANSFIERE. El hallazgo del Exp. 2d se reproduce en notas: la forma del")
        print(f"    nombre COMPLEMENTA al texto ({h4_v}). Pero con techo {100*techo:.2f} %:")
        print( "    la capa solo puede tocar las 64 notas con nombre auditado.")
    elif mejor_add == mejor_add and mejor_add >= 0.80:
        print(f"    NO MUEVE EL AGREGADO, PERO ACIERTA. La mejor ({mejor}) acierta")
        print(f"    {mejor_add:.4f} donde agrega cobertura, pero solo agrega "
              f"{df[df.variante==mejor].cobertura_agregada.iloc[0]:.4f} del corpus.")
        print( "    Mismo veredicto que el POSTHOC de LOGO: regla de alta precision y baja")
        print( "    cobertura que no mueve el agregado. NO SE ADOPTA, y la razon es el techo")
        print( "    del corpus (42,95 %), no la abstraccion.")
    else:
        print( "    LA TRANSFERENCIA FALLA. La forma del nombre NO es firma de familia en notas.")
        print(f"    La mejor ({mejor}) acierta {mejor_add} donde agrega cobertura.")
        print( "    Explicacion coherente con los dos frentes: en NapierOne cada familia es UNA")
        print( "    campana -- la forma del nombre ES la campana, que es la limitacion ya")
        print( "    declarada del Exp. 2d -- mientras que en el corpus de notas cada familia trae")
        print( "    notas de varias campanas, que renombran distinto.")

    # ---------------- VEREDICTO DE LA AMPLIACION (2026-09-28) ----------------
    print("\n" + "=" * 90)
    print("  VEREDICTO DE LA AMPLIACION -- los dos grupos por separado (pedido del 2026-09-28)")
    print("=" * 90)
    print("  GRUPO A = solo lo que pone el ransomware (inmune a la mascara del auditor)")
    print("  GRUPO B = mira el nombre ENTERO: largo literal, cantidad de tokens, tipografia\n")
    print(f"  {'variante':<10} {'grupo':<14} {'D macro-F1 global':>20} {'IC 95 %':>24} {'sem':>7}")
    for v in GRUPO_A + GRUPO_B + ["GRUPO_A", "GRUPO_B", "TODAS"]:
        d, lo, hi, sp = getd(v, "macro_f1_global")
        g = ("A solida" if v in GRUPO_A else "B sospechosa" if v in GRUPO_B else "combo")
        print(f"  {v:<10} {g:<14} {d:>+20.4f} {'[%+.4f; %+.4f]' % (lo, hi):>24} {sp:>4}/{S}")

    d_rob, lo_rob, _, _ = getd("ROB", "macro_f1_global")
    d_frob, lo_frob, _, _ = getd("FROB", "macro_f1_global")
    d_ga, lo_ga, hi_ga, _ = getd("GRUPO_A", "macro_f1_global")
    d_gb, lo_gb, hi_gb, _ = getd("GRUPO_B", "macro_f1_global")
    d_firma = getd("FIRMA", "macro_f1_global")[0]
    a1 = (d_rob <= 0 or lo_rob <= 0) and (d_frob <= 0 or lo_frob <= 0)
    # A2: de las candidatas del preregistro, cuales dan Delta > 0 con IC que excluye el cero
    positivas = [v for v in CANDIDATAS if getd(v, "macro_f1_global")[1] > 0]
    a2 = bool(positivas) and all(v in GRUPO_B for v in positivas)
    a3 = d_ga <= 0
    a5 = d_frob >= d_firma
    amp = [
        ("A1 [ciega] ROB y FROB no aportan (D <= 0 o IC incluye 0)", a1,
         f"ROB {d_rob:+.4f} · FROB {d_frob:+.4f}"),
        ("A2 [observacion] el unico aporte nominal es del GRUPO B", a2,
         f"positivas: {positivas if positivas else 'ninguna'}"),
        ("A3 [ciega] GRUPO_A combinado da D <= 0", a3, f"{d_ga:+.4f} [{lo_ga:+.4f}; {hi_ga:+.4f}]"),
        ("A4 [ciega] hay >= 2 notaciones de la mascara del ID", ok_a4,
         f"{len(masc)} notaciones · {len(cambia)} firmas cambian"),
        ("A5 [ciega] FROB no da menos que FIRMA", a5,
         f"FROB {d_frob:+.4f} vs FIRMA {d_firma:+.4f}"),
    ]
    for nombre, cumple, det in amp:
        print(f"\n  [{'CUMPLE' if cumple else 'FALLA '}] {nombre:<56} {det}")

    print("\n" + "-" * 90)
    print("  POR QUE LA LECTURA AUTOMATICA DE H10 ES DEMASIADO GENEROSA")
    print("-" * 90)
    print("  H10 se escribio mirando SOLO si algun Delta macro-F1 global tenia IC que excluye el")
    print("  cero. Esa condicion se puede cumplir sin que la capa sirva, y aca pasa exactamente")
    print("  eso. Los numeros de la unica variante que la cumple:")
    r_esq = df[df.variante == "ESQ"].iloc[0]
    d_e_ac, lo_e_ac, hi_e_ac, sp_e_ac = getd("ESQ", "exactitud_global")
    d_e_fr, lo_e_fr, hi_e_fr, _ = getd("ESQ", "macro_f1_restringido")
    print(f"    - acierto donde AGREGA cobertura ....... {r_esq['acierto_en_lo_agregado']}"
          "   (la capa se equivoca la mayoria de las veces que dispara de mas)")
    print(f"    - cobertura que ROMPE .................. {r_esq['cobertura_rota']}"
          f"   y ahi la base acertaba {r_esq['acierto_base_en_lo_roto']}")
    print(f"    - Delta EXACTITUD global ............... {d_e_ac:+.4f} "
          f"[{lo_e_ac:+.4f}; {hi_e_ac:+.4f}]  {sp_e_ac}/{S}  (el IC incluye el cero)")
    print(f"    - Delta macro-F1 RESTRINGIDO a las 64 .. {d_e_fr:+.4f} "
          f"[{lo_e_fr:+.4f}; {hi_e_fr:+.4f}]  (NEGATIVO donde la capa puede actuar)")
    movidas = dff[(dff.n_con_nombre > 0) & (dff.delta.abs() > 1e-9)]
    print(f"    - familias que se mueven: {len(movidas)} de {len(fam_con_nombre)} con nombre")
    for _, r in movidas.iterrows():
        print(f"        {r['familia']:<14} {r['base']:.4f} -> {r['mejor']:.4f}  ({r['delta']:+.4f})"
              f"  [{r['n_notas']} notas, {r['n_con_nombre']} con nombre]")
    print("\n  Traduccion: el +0,0042 de macro-F1 global de ESQ no es una mejora del sistema.")
    print("  Es el macro-F1 reaccionando al movimiento de una familia chica, mientras la")
    print("  exactitud no se mueve y el macro-F1 restringido BAJA. Y ademas ESQ es del GRUPO B,")
    print("  el sospechoso. Por eso la lectura que vale es la del segundo/tercer caso de H10 y")
    print("  NO la que imprimio la regla automatica.")

    print("\n  RECORDAR AL CITAR: toda cifra global de este script esta multiplicada por 64/149.")
    print(f"  El aporte RESTRINGIDO a las notas con nombre es {fac:.3f} veces el global en exactitud.")
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
