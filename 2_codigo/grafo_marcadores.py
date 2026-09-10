#!/usr/bin/env python3
"""
grafo_marcadores.py -- B.3: grafo de marcadores compartidos entre plantillas + protocolo P3.

QUE CONTESTA
1. Elemento de accion 1 del tutor (reunion 2026-08-12): "hallar algun patron de
   aprendizaje entre plantillas para poder detectar una no conocida".
2. La pregunta que quedo abierta al cerrar B.1: 12 familias tienen exactamente 2
   plantillas y su F1 POR FAMILIA va de 0,000 (CHIMERA) a 1,000 (SUNCRYPT). La cantidad
   de plantillas NO explica el desempeno. Lo que falta medir es CUANTO SE PARECEN entre
   si las plantillas de una misma familia -- y eso decide cuales de las 9 familias de la
   lista de recoleccion van a responder a mas textos y cuales no.
3. Diagnostico previo de C.bis: si las plantillas de una familia casi no comparten
   valores de marcador, la vista de marcadores no puede funcionar bajo P2, y se sabe
   ANTES de escribirla.

EL GRAFO
  Nodos  = las 95 plantillas (componentes de casi-duplicados, coseno char 3-5 > 0,90).
  Aristas = dos plantillas comparten el VALOR EXACTO de un marcador.
  Marcadores: los mismos patrones declarados en normalizacion_marcadores.py (ONION,
  EMAIL, BTC, URL, ID, CLAVE). Se reusan a proposito: es el criterio ya declarado en la
  tesis y no se inventa uno nuevo para este experimento.

PROTOCOLO P3, Y POR QUE LA DIFERENCIA P2 - P3 NO SE PUEDE LEER SOLA
El grupo del StratifiedGroupKFold pasa a ser la COMPONENTE CONEXA del grafo, en vez de
la plantilla. Asi el modelo nunca ve un IOC del conjunto de prueba.

  !! La diferencia P2 - P3 NO mide por si sola la continuidad de IOCs. Bajo P3 los grupos
  bajan de 95 a 65, con lo cual hay 7 familias que quedan enteras en un solo pliegue
  (contra 4,2 en P2; su F1 es 0 por construccion) y cada pliegue entrena con menos
  unidades independientes. Esa segunda causa esta mezclada con la primera.

  Por eso hay CONTROL DE AZAR: agrupamientos al azar con el mismo perfil de tamanos por
  familia que las componentes reales, pero eligiendo al azar que plantillas caen juntas.
  Aisla "cuantas plantillas se fusionaron" de "se fusionaron JUSTO las que comparten IOCs".

  RESULTADO MEDIDO (2026-08-19, 144 notas, n=20 al azar): P2 0,4210 · azar 0,2251 +/- 0,0149 · P3 real
  0,2293 (percentil 65 de la nube de azar, t = -1,26, p = 0,223). La caida total de 0,1917 se
  descompone en 0,1959 de agrupamiento grueso y -0,0042 de continuidad de IOCs (signo
  NEGATIVO: agrupar por IOCs lastima MENOS que el azar). => El macro-F1 de P2 NO viene de
  reconocer IOCs repetidos. Es la respuesta a la objecion "tu 0,435
  es busqueda de IOCs disfrazada", que es justo la que se espera del tutor, que conoce a
  Lemmou et al. (2021) -- cuyo metodo de identificacion de familia SI es busqueda de
  casi-duplicados por reglas y marcadores.

  CONSECUENCIA: P3 no reemplaza ni mejora a P2. Es el mismo protocolo con agrupamiento mas
  grueso, y por lo tanto mas ruidoso. En la tesis se reporta P2; P3 entra como el control
  que descarta la contaminacion por IOCs, no como un tercer protocolo.

DOS EJES DE VARIANTE, LOS CUATRO SE CORREN Y SE REPORTAN
(a) Circularidad -- "la feature sobreviviria si la familia se cambiara el nombre manana?"
    Se excluye todo valor de marcador que contenga el nombre de la familia como
    subcadena (minusculas, en cualquier posicion: parte local, dominio, onion, ruta) mas
    los alias conocidos. `lockbitsupp@...` afuera; `abc123@protonmail.com` adentro.
    Se corre CON y SIN la exclusion y se reportan las dos.
    OJO CON LA SUBCADENA: "conti" esta dentro de "continue" y "cerber" dentro de
    "cerberus". El criterio se aplica solo a valores de marcador (no a texto libre), pero
    el riesgo existe: por eso se emite la lista completa de valores excluidos, para poder
    auditarla a mano.
(b) Valores genericos -- un mismo valor puede ser infraestructura comun y no una firma de
    campana (torproject.org, tox.chat, una casa de cambio). Si un valor aparece en muchas
    familias, une el grafo entero en una sola componente y destruye P3. Se corre sin
    filtro y filtrando los valores que aparecen en mas de MAX_FAMILIAS_VALOR familias, y
    se emite `b3_valores_compartidos.csv` ordenado por familias, para ver quien pesa.

Uso:
    python grafo_marcadores.py                 # todo
    python grafo_marcadores.py --solo-grafo    # sin evaluar P1/P2/P3 (rapido)

Salidas en 4_resultados/resultados_grafo_marcadores/. Los promedios entre semillas los
calcula resumen_para_capitulo4.py --solo b3, no este script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))

from clasificador_notas_v2 import (CORPUS_DIR, N_FOLDS, N_SEMILLAS, TFIDF_CHAR,
                                   UMBRAL_NEARDUP, agrupar_neardups, cargar_corpus,
                                   evaluar)
from normalizacion_marcadores import PATRONES

OUT_DIR = (_AQUI.parent / "4_resultados" / "resultados_grafo_marcadores"
           if (_AQUI.parent / "4_resultados").is_dir()
           else _AQUI / "resultados_grafo_marcadores")

# Un valor presente en mas de estas familias se considera infraestructura comun, no firma
# de campana. El 2 deja pasar los parentescos entre dos familias (BLACKBASTA/CONTI,
# DHARMA/PHOBOS), que son justamente los interesantes.
MAX_FAMILIAS_VALOR = 2

# Alias conocidos, para el criterio de circularidad. Clave = familia del corpus.
ALIAS = {
    "BLACKCAT": ["alphv", "alpha spider"],
    "SODINOKIBI": ["revil", "sodin"],
    "MEDUZALOCKER": ["medusalocker", "medusa"],
    "BLACKBASTA": ["basta"],
    "WASTEDLOCKER": ["wasted"],
}

# Los URL se comparan sin puntuacion final ni fragmento; los mails y onion en minuscula.
# El BTC NO se pasa a minuscula: base58 distingue mayusculas y bajarlas corrompe la
# direccion y podria fusionar dos billeteras distintas.
_SENSIBLE_A_MAYUSCULAS = {"[BTC]", "[ID]", "[CLAVE]"}


def _normalizar_valor(etiqueta, valor):
    v = valor.strip().rstrip(".,;:)»\"'>]")
    if etiqueta not in _SENSIBLE_A_MAYUSCULAS:
        v = v.lower()
    return v


def extraer_marcadores(texto):
    """[(tipo, valor_normalizado)] de una nota. Mismos patrones que el Sprint 1.1."""
    fuera = []
    resto = texto
    for etiqueta, rx in PATRONES:
        for m in rx.finditer(resto):
            fuera.append((etiqueta, _normalizar_valor(etiqueta, m.group(0))))
        # Igual que en normalizacion_marcadores: el orden importa y cada patron consume
        # su parte, para que un mail dentro de una URL no se cuente dos veces.
        resto = rx.sub(" ", resto)
    return fuera


def _terminos_circulares(familia):
    t = {familia.lower()}
    t.update(a.lower() for a in ALIAS.get(familia, []))
    return {x for x in t if len(x) >= 4}


def construir_grafo(marcadores_por_plantilla, fam_de_plantilla, *,
                    excluir_nombre_familia, max_familias_valor):
    """Devuelve (aristas, componentes, excluidos, valores).

    `marcadores_por_plantilla`: {plantilla: {(tipo, valor)}}
    `componentes`: {plantilla: id_de_componente} por union-find.
    """
    # 1. valor -> plantillas que lo tienen, y familias que lo tienen
    valores = defaultdict(set)
    excluidos = []
    for pl, ms in marcadores_por_plantilla.items():
        fam = fam_de_plantilla[pl]
        for tipo, val in ms:
            if excluir_nombre_familia and any(t in val for t in _terminos_circulares(fam)):
                excluidos.append(dict(plantilla=pl, familia=fam, tipo=tipo, valor=val))
                continue
            valores[(tipo, val)].add(pl)

    genericos = set()
    if max_familias_valor:
        for clave, pls in valores.items():
            if len({fam_de_plantilla[p] for p in pls}) > max_familias_valor:
                genericos.add(clave)

    # 2. aristas
    aristas = []
    for (tipo, val), pls in sorted(valores.items()):
        if (tipo, val) in genericos or len(pls) < 2:
            continue
        for a, b in combinations(sorted(pls), 2):
            aristas.append(dict(plantilla_a=a, plantilla_b=b, tipo=tipo, valor=val,
                                misma_familia=fam_de_plantilla[a] == fam_de_plantilla[b]))

    # 3. componentes conexas (union-find sobre las plantillas)
    padre = {p: p for p in marcadores_por_plantilla}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for e in aristas:
        ra, rb = raiz(e["plantilla_a"]), raiz(e["plantilla_b"])
        if ra != rb:
            padre[rb] = ra
    componentes = {p: raiz(p) for p in padre}
    return aristas, componentes, excluidos, valores, genericos


def cohesion_por_familia(fams, plantillas, centroides, marcadores_por_plantilla,
                         aristas_misma_familia):
    """Para cada familia: cuanto se parecen sus plantillas entre si.

    Es la salida que decide la recoleccion. Dos medidas independientes:
      - `coseno_medio_entre_plantillas`: similitud de texto entre pares de plantillas de
        la familia (por construccion todas por DEBAJO de 0,90, que es el umbral de
        casi-duplicado). Mide parecido linguistico.
      - `pares_unidos_por_marcador` / `pares_totales`: fraccion de pares de plantillas de
        la familia que comparten al menos un valor de marcador. Mide continuidad de IOCs.
    """
    porfam = defaultdict(list)
    for pl in plantillas:
        porfam[fams[pl]].append(pl)
    unidos = Counter()
    for e in aristas_misma_familia:
        a, b = e["plantilla_a"], e["plantilla_b"]
        unidos[(fams[a], tuple(sorted((a, b))))] = 1
    pares_unidos = Counter()
    for (fam, _par) in unidos:
        pares_unidos[fam] += 1

    filas = []
    for fam, pls in sorted(porfam.items()):
        pares = list(combinations(sorted(pls), 2))
        if pares:
            sims = [float(centroides[a] @ centroides[b]) for a, b in pares]
            cos_medio, cos_max, cos_min = np.mean(sims), max(sims), min(sims)
        else:
            cos_medio = cos_max = cos_min = float("nan")
        # La cohesion interna sola NO alcanza para predecir el F1 de una familia: tambien
        # importa cuanto se parecen sus plantillas a las de OTRAS familias. Una familia
        # con plantillas poco parecidas entre si puede acertar igual si ninguna otra
        # familia se le parece (caso SUNCRYPT). El separador honesto es el MARGEN.
        ajenas = [p for p in plantillas if fams[p] != fam]
        inter = [float(centroides[a] @ centroides[b]) for a in pls for b in ajenas]
        cos_inter_max = max(inter) if inter else float("nan")
        cos_inter_medio = float(np.mean(inter)) if inter else float("nan")
        n_marc = sum(len(marcadores_por_plantilla[p]) for p in pls)
        tipos = Counter(t for p in pls for t, _ in marcadores_por_plantilla[p])
        filas.append(dict(
            familia=fam, n_plantillas=len(pls), pares_totales=len(pares),
            coseno_medio_entre_plantillas=round(cos_medio, 4),
            coseno_max=round(cos_max, 4), coseno_min=round(cos_min, 4),
            coseno_max_a_otra_familia=round(cos_inter_max, 4),
            coseno_medio_a_otra_familia=round(cos_inter_medio, 4),
            # margen > 0: las plantillas de la familia se parecen mas entre si que a
            # cualquier plantilla ajena => hay algo propio que aprender.
            margen=round(cos_medio - cos_inter_max, 4),
            pares_unidos_por_marcador=pares_unidos.get(fam, 0),
            fraccion_pares_unidos=(round(pares_unidos.get(fam, 0) / len(pares), 4)
                                   if pares else float("nan")),
            marcadores_totales=n_marc,
            **{f"n_{t.strip('[]').lower()}": tipos.get(t, 0) for t, _ in PATRONES}))
    return pd.DataFrame(filas)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-grafo", action="store_true",
                    help="no evaluar P1/P2/P3, solo construir el grafo y la cohesion")
    ap.add_argument("--max-familias-valor", type=int, default=MAX_FAMILIAS_VALOR)
    ap.add_argument("--repeticiones-azar", type=int, default=5,
                    help="agrupamientos al azar del control (mismo perfil de tamanos)")
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (por defecto, 4_resultados/resultados_grafo_marcadores). "
                         "Usar una carpeta NUEVA para no pisar la corrida canonica.")
    args = ap.parse_args()
    if args.salida is not None:
        OUT_DIR = args.salida

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  B.3 -- GRAFO DE MARCADORES COMPARTIDOS ENTRE PLANTILLAS")
    print("=" * 78)
    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    grupos, _ = agrupar_neardups(textos, UMBRAL_NEARDUP)
    familias = np.unique(y)
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Notas: {len(textos)} | Familias: {len(familias)} | "
          f"Plantillas: {len(set(grupos))}")

    # ---- nodos: la plantilla es el par (familia, componente de casi-duplicado), porque
    # dos componentes contienen notas de dos familias distintas (grupo 6 = BLACKBASTA +
    # CONTI; grupo 55 = DHARMA + PHOBOS) y hay que poder distinguirlas.
    marcadores = defaultdict(set)
    fam_de_plantilla, notas_de_plantilla = {}, defaultdict(list)
    for i, (t, fam, g) in enumerate(zip(textos, y, grupos)):
        pl = f"{fam}#{g}"
        fam_de_plantilla[pl] = fam
        notas_de_plantilla[pl].append(i)
        marcadores[pl].update(extraer_marcadores(t))
    plantillas = sorted(marcadores)
    print(f"Nodos del grafo (familia#plantilla): {len(plantillas)}")

    tot = Counter(t for ms in marcadores.values() for t, _ in ms)
    print("Marcadores hallados por tipo: " +
          " · ".join(f"{k.strip('[]')} {v}" for k, v in sorted(tot.items())))
    sin_marcador = [p for p in plantillas if not marcadores[p]]
    print(f"Plantillas SIN ningun marcador: {len(sin_marcador)} de {len(plantillas)}")

    pd.DataFrame([dict(plantilla=p, familia=fam_de_plantilla[p],
                       n_notas=len(notas_de_plantilla[p]),
                       n_marcadores=len(marcadores[p]),
                       tipos="|".join(sorted({t for t, _ in marcadores[p]})),
                       valores="|".join(sorted(v for _, v in marcadores[p])))
                  for p in plantillas]).to_csv(
        OUT_DIR / "b3_marcadores_por_plantilla.csv", index=False)

    # ---- centroides de texto por plantilla, para la cohesion linguistica
    X = TfidfVectorizer(**TFIDF_CHAR).fit_transform(textos)
    cent = {}
    for p in plantillas:
        v = np.asarray(X[notas_de_plantilla[p]].mean(axis=0)).ravel()
        n = np.linalg.norm(v)
        cent[p] = v / n if n else v

    # ---- las cuatro variantes del grafo
    filas_comp, filas_ar, resumen_var, filas_exc = [], [], [], []
    componentes_por_variante = {}
    for excluir in (False, True):
        for maxfam in (0, args.max_familias_valor):
            etq = ("sin_exclusion" if not excluir else "con_exclusion") + \
                  ("_sin_filtro" if not maxfam else f"_filtro{maxfam}")
            aristas, comp, exc, valores, genericos = construir_grafo(
                marcadores, fam_de_plantilla,
                excluir_nombre_familia=excluir, max_familias_valor=maxfam)
            tam = Counter(comp.values())
            n_comp = len(tam)
            mayor = max(tam.values())
            mismas = sum(1 for e in aristas if e["misma_familia"])
            # familias que quedan con una sola componente => inevaluables bajo P3
            comp_por_fam = defaultdict(set)
            for p, c in comp.items():
                comp_por_fam[fam_de_plantilla[p]].add(c)
            inevaluables = sorted(f for f, cs in comp_por_fam.items() if len(cs) < 2)
            resumen_var.append(dict(
                variante=etq, excluye_nombre_familia=excluir,
                max_familias_por_valor=maxfam, n_aristas=len(aristas),
                aristas_misma_familia=mismas,
                aristas_entre_familias=len(aristas) - mismas,
                n_componentes=n_comp, componente_mayor=mayor,
                valores_genericos_excluidos=len(genericos),
                valores_excluidos_por_nombre=len(exc),
                familias_inevaluables_p3=len(inevaluables),
                lista_inevaluables="|".join(inevaluables)))
            componentes_por_variante[etq] = comp
            for p, c in sorted(comp.items()):
                filas_comp.append(dict(variante=etq, plantilla=p,
                                       familia=fam_de_plantilla[p], componente=c))
            for e in aristas:
                filas_ar.append(dict(variante=etq, **e))
            for e in exc:
                filas_exc.append(dict(variante=etq, **e))
            print(f"\n[{etq}] aristas {len(aristas)} "
                  f"({mismas} dentro de familia, {len(aristas) - mismas} entre familias) | "
                  f"componentes {n_comp} | mayor {mayor} | "
                  f"familias inevaluables bajo P3 {len(inevaluables)}")
            if inevaluables:
                print(f"    inevaluables: {', '.join(inevaluables)}")

    pd.DataFrame(resumen_var).to_csv(OUT_DIR / "b3_variantes.csv", index=False)
    pd.DataFrame(filas_comp).to_csv(OUT_DIR / "b3_componentes.csv", index=False)
    pd.DataFrame(filas_ar).to_csv(OUT_DIR / "b3_aristas.csv", index=False)
    pd.DataFrame(filas_exc).to_csv(OUT_DIR / "b3_valores_excluidos.csv", index=False)

    # ---- que valores unen mas: para auditar el filtro de genericos a mano
    _, _, _, valores_todos, _ = construir_grafo(
        marcadores, fam_de_plantilla, excluir_nombre_familia=False,
        max_familias_valor=0)
    pd.DataFrame(sorted(
        (dict(tipo=t, valor=v, n_plantillas=len(pls),
              n_familias=len({fam_de_plantilla[p] for p in pls}),
              familias="|".join(sorted({fam_de_plantilla[p] for p in pls})))
         for (t, v), pls in valores_todos.items() if len(pls) > 1),
        key=lambda d: (-d["n_familias"], -d["n_plantillas"]))).to_csv(
        OUT_DIR / "b3_valores_compartidos.csv", index=False)

    # ---- cohesion por familia: la salida que decide la recoleccion
    aristas_mf = [e for e in filas_ar
                  if e["variante"] == f"sin_exclusion_filtro{args.max_familias_valor}"
                  and e["misma_familia"]]
    coh = cohesion_por_familia(fam_de_plantilla, plantillas, cent, marcadores, aristas_mf)
    coh.to_csv(OUT_DIR / "b3_cohesion_por_familia.csv", index=False)
    print("\n" + "-" * 78)
    print("  COHESION POR FAMILIA -- cuanto se parecen sus plantillas entre si")
    print("  (coseno char 3-5 entre centroides de plantilla; por construccion < 0,90)")
    print("-" * 78)
    print(coh.sort_values("margen")[
        ["familia", "n_plantillas", "coseno_medio_entre_plantillas",
         "coseno_max_a_otra_familia", "margen", "fraccion_pares_unidos",
         "marcadores_totales"]].to_string(index=False))

    # ---- CONTROL DE AZAR: sin esto el numero de P3 no es defendible
    # Bajo P3 los grupos pasan de 95 a 43-65. Parte de la caida NO es "el modelo ya no
    # puede aprovechar la continuidad de IOCs" (el efecto buscado) sino simplemente que el
    # agrupamiento es mas grueso: hay mas familias que quedan enteras en un solo pliegue y
    # cada pliegue tiene menos unidades independientes para entrenar.
    # El control separa las dos causas: se reparten las plantillas de CADA familia en
    # bloques con el MISMO perfil de tamanos que las componentes reales, pero eligiendo al
    # azar cuales caen juntas. Si el azar cae lo mismo que P3, la caida es del agrupamiento
    # grueso y no de los IOCs. La diferencia P3 - azar es el efecto real de los IOCs.
    # Simplificacion declarada: las pocas componentes que cruzan familias se reparten
    # dentro de cada familia por separado.
    def grupos_al_azar(comp, semilla):
        perfil = defaultdict(list)
        for p, c in comp.items():
            perfil[fam_de_plantilla[p]].append(c)
        rng = np.random.default_rng(semilla)
        asignado = {}
        for fam, cs in perfil.items():
            tam = sorted(Counter(cs).values(), reverse=True)
            pls = sorted(p for p in comp if fam_de_plantilla[p] == fam)
            pls = [pls[i] for i in rng.permutation(len(pls))]
            i = 0
            for j, t in enumerate(tam):
                for p in pls[i:i + t]:
                    asignado[p] = f"{fam}#azar{j}"
                i += t
        return asignado

    # ---- P1 / P2 / P3
    filas_prot = []
    if not args.solo_grafo:
        print("\n" + "-" * 78)
        print("  PROTOCOLOS -- P1, P2 y P3 sobre el mismo corpus")
        print(f"  (combinado + LinearSVC, {N_FOLDS} pliegues, {N_SEMILLAS} semillas)")
        print("-" * 78)
        tareas = [("P1", "estratificado", "caracteres", grupos),
                  ("P2", "grupos", "combinado", grupos)]
        for etq, comp in componentes_por_variante.items():
            g3 = np.array([comp[f"{fam}#{g}"] for fam, g in zip(y, grupos)])
            tareas.append((f"P3:{etq}", "grupos", "combinado", g3))
        for etq, prot, vista, gg in tareas:
            res, (p, r, f), _ = evaluar(textos, y, gg, vista, "LinearSVC", prot, familias)
            filas_prot.append(dict(protocolo=etq, vista=vista,
                                   n_grupos=int(len(set(gg))), **res))
            print(f"  {etq:<28} grupos {len(set(gg)):>3} | "
                  f"macro-F1 {res['f1_macro_mean']:.4f} ± {res['f1_macro_std']:.4f} | "
                  f"exactitud {res['accuracy_mean']:.4f} | "
                  f"bal.acc {res['balanced_accuracy_mean']:.4f}")

        # Control de azar sobre la variante mas estricta y mas defendible
        etq_ref = f"con_exclusion_filtro{args.max_familias_valor}"
        print(f"\n  CONTROL DE AZAR sobre {etq_ref} "
              f"({args.repeticiones_azar} agrupamientos con el mismo perfil de tamanos):")
        f1s = []
        for s in range(args.repeticiones_azar):
            az = grupos_al_azar(componentes_por_variante[etq_ref], 500 + s)
            gz = np.array([az[f"{fam}#{g}"] for fam, g in zip(y, grupos)])
            res, _, _ = evaluar(textos, y, gz, "combinado", "LinearSVC", "grupos", familias)
            f1s.append(res["f1_macro_mean"])
            filas_prot.append(dict(protocolo=f"azar:{etq_ref}#{s}", vista="combinado",
                                   n_grupos=int(len(set(gz))), **res))
            print(f"    azar {s}: grupos {len(set(gz)):>3} | "
                  f"macro-F1 {res['f1_macro_mean']:.4f}")
        real = [r for r in filas_prot if r["protocolo"] == f"P3:{etq_ref}"][0]
        p2 = [r for r in filas_prot if r["protocolo"] == "P2"][0]
        m_az = float(np.mean(f1s))
        d_az = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0
        print(f"\n    P2                     macro-F1 {p2['f1_macro_mean']:.4f}")
        print(f"    azar (mismo perfil)    macro-F1 {m_az:.4f} ± {d_az:.4f}")
        print(f"    P3 real                macro-F1 {real['f1_macro_mean']:.4f}")
        print(f"    => caida total P2-P3        {p2['f1_macro_mean'] - real['f1_macro_mean']:+.4f}")
        print(f"       de eso, agrupamiento grueso {p2['f1_macro_mean'] - m_az:+.4f}")
        print(f"       y continuidad de IOCs       {m_az - real['f1_macro_mean']:+.4f}")
        pd.DataFrame(filas_prot).to_csv(OUT_DIR / "b3_protocolos.csv", index=False)

    (OUT_DIR / "manifiesto_b3.json").write_text(json.dumps(dict(
        fecha=str(date.today()), corpus=str(CORPUS_DIR), n_notas=len(textos),
        n_familias=int(len(familias)), n_plantillas=int(len(plantillas)),
        umbral_neardup=UMBRAL_NEARDUP, max_familias_valor=args.max_familias_valor,
        patrones={e: rx.pattern for e, rx in PATRONES},
        alias=ALIAS, marcadores_por_tipo=dict(tot),
        plantillas_sin_marcador=len(sin_marcador),
        n_folds=N_FOLDS, n_semillas=N_SEMILLAS,
        sklearn=sklearn.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSalidas en: {OUT_DIR}")
    print("Los promedios y las tablas: python resumen_para_capitulo4.py --solo b3")


if __name__ == "__main__":
    main()
