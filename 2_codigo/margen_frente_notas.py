"""Donde queda margen en el frente de notas: MAS NOTAS o OTRO CANAL. Medido, no supuesto.

PREGUNTA QUE CONTESTA
  Con B.1 re-medida sobre 149 (el corte bajo de 4 a 3 plantillas) y con M.6 adoptado
  (+0,0599), la pregunta operativa dejo de ser "cuantas notas faltan" y paso a ser "que
  tipo de dato falta". Este script cruza, por familia, las tres cosas que deciden eso:
    1. cuanto material tiene (notas, plantillas) y como rinde (F1 base y F1 M.6),
    2. cuanto la cubre la regla de M.6 (IOCs privados + nombre genuino) y con que acierto,
       y como le va al texto solo en las notas que la regla NO cubre,
    3. cuanto gano de verdad cada familia por tener una plantilla mas, separando el efecto
       PROPIO del AJENO (ver abajo, es la trampa principal).

LA TRAMPA DEL EFECTO PROPIO VS AJENO
  El tope k de la curva B.1 se aplica a TODAS las familias a la vez. Para una familia con
  n plantillas y un paso k -> k+1:
    - si n >= k+1, el paso le agrega una plantilla PROPIA: el Delta mide lo que gana ESA
      familia con mas dato SUYO. Es lo que hay que mirar para decidir si conviene salir a
      recolectar para ella.
    - si n <= k, el tope no le muerde ni antes ni despues: su entrenamiento es IDENTICO y
      el Delta mide solo el efecto AJENO (que las OTRAS familias tengan mas dato).
  Leer el segundo como si fuera el primero da conclusiones al reves: una familia de 2
  plantillas que "sube" en el paso 2->3 no esta diciendo que convenga recolectar para ella
  -- esta diciendo que le hace bien que las demas esten mejor cubiertas.

COTA DEL CANAL (declarada, no predicha)
  Para las familias que la regla no cubre se calcula una COTA SUPERIOR: cuanto daria el
  macro-F1 si esas familias alcanzaran el F1 medio de las que hoy si estan bien cubiertas.
  Es un supuesto explicito y falsable, y se reporta como cota -- igual que la tercera
  columna del Exp. 2d. NO es una prediccion de lo que va a pasar.

Uso:
    python margen_frente_notas.py
    python margen_frente_notas.py --salida ../4_resultados/margen_notas_149 \
        --curva ../4_resultados/resultados_curva_149 \
        --m6 ../4_resultados/resultados_cascada_combinada_149 \
        --techo ../4_resultados/resultados_techo_por_familia_149
Todas las entradas son parametros: NINGUNA ruta de resultados esta fija en el codigo, y el
manifiesto registra de donde salio cada cifra. (Es el patron que se arreglo en cinco
scripts el 2026-08-25: la base escrita a mano fue la fuente de todos los errores caros.)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
_RES = _AQUI.parent / "4_resultados"

VARIANTE_M6 = "privados_sin_circ_MAS_NOMBRE"   # la variante de M.6 que se reporta
CURVA_CLAVE = ("30fam", "P2ret", "plantillas")  # la unica con plantillas por familia
UMBRAL_COB_ALTA = 0.50                          # que cuenta como "bien cubierta"


def leer_techo(d: Path):
    """Plantillas, notas, cohesion y F1 por familia (salida de techo_por_familia.py)."""
    f = d / "techo_por_familia.csv"
    if not f.exists():
        sys.exit(f"ABORTA: falta {f}. Correr primero techo_por_familia.py.")
    info = {}
    with open(f, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            coh = r["coseno_medio_entre_plantillas"]
            info[r["familia"]] = dict(
                notas=int(r["n_notas"]), plantillas=int(r["n_plantillas"]),
                # Una familia de 1 plantilla no tiene pares: la cohesion NO existe, no es 0.
                cohesion=float(coh) if coh else float("nan"),
                f1_base=float(r["f1_base"]), f1_m6=float(r["f1_m6"]))
    return info


def leer_cobertura(d: Path, variante: str):
    """Cobertura de la regla de M.6 por familia, y acierto del texto donde NO cubre."""
    f = d / "m6_diagnostico_nota_por_nota.csv"
    if not f.exists():
        sys.exit(f"ABORTA: falta {f}. Correr primero cascada_combinada_notas.py.")
    tot, apl, nom, ok_r, no_cub, ok_t = (defaultdict(int) for _ in range(6))
    vistas = set()
    with open(f, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            vistas.add(r["variante"])
            if r["variante"] != variante:
                continue
            fam = r["familia_real"]
            tot[fam] += 1
            if r["estado"] == "asignada":
                apl[fam] += 1
                ok_r[fam] += r["acierta_cascada"] == "True"
                nom[fam] += r["uso_nombre"] == "True"
            else:
                no_cub[fam] += 1
                ok_t[fam] += r["acierta_texto"] == "True"
    if not tot:
        sys.exit(f"ABORTA: la variante '{variante}' no esta en {f.name}. "
                 f"Variantes presentes: {sorted(vistas)}")
    out = {}
    for fam in tot:
        out[fam] = dict(
            cobertura=apl[fam] / tot[fam],
            cobertura_por_nombre=nom[fam] / tot[fam],
            acierto_regla=(ok_r[fam] / apl[fam]) if apl[fam] else float("nan"),
            acierto_texto_no_cubiertas=(ok_t[fam] / no_cub[fam]) if no_cub[fam]
            else float("nan"))
    return out, sorted(vistas)


def deltas_por_familia(curva_dir: Path, info):
    """Delta pareado de F1 por familia en cada paso k->k+1, etiquetado propio/ajeno."""
    f = curva_dir / "b1_curva_por_familia.csv"
    if not f.exists():
        sys.exit(f"ABORTA: falta {f}. Correr primero curva_aprendizaje_notas.py.")
    por = defaultdict(lambda: defaultdict(dict))
    with open(f, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if (r["curva"], r["protocolo"], r["unidad"]) != CURVA_CLAVE:
                continue
            if r["k"].isdigit():
                por[r["familia"]][int(r["k"])][r["repeticion"]] = float(r["f1"])
    filas = []
    for fam in sorted(por):
        n = info.get(fam, {}).get("plantillas", 0)
        ks = sorted(por[fam])
        for k1, k2 in zip(ks, ks[1:]):
            a, b = por[fam][k1], por[fam][k2]
            reps = sorted(set(a) & set(b))
            if len(reps) < 2:
                continue
            d = np.array([b[i] - a[i] for i in reps])
            s = d.std(ddof=1)
            ee = s / np.sqrt(len(d)) if s > 0 else 0.0
            t = stats.t.ppf(0.975, len(d) - 1)
            inf, sup = d.mean() - t * ee, d.mean() + t * ee
            # el tope le muerde a ESTA familia solo si tiene con que llegar a k2
            propio = n >= k2
            filas.append(dict(
                familia=fam, paso=f"{k1}->{k2}", k_desde=k1, k_hasta=k2,
                efecto="propio" if propio else "ajeno",
                plantillas=n, cohesion=info.get(fam, {}).get("cohesion", float("nan")),
                delta=d.mean(), ic95_inf=inf, ic95_sup=sup, n_reps=len(d),
                veredicto="SUBE" if inf > 0 else ("BAJA" if sup < 0 else "sin efecto")))
    return pd.DataFrame(filas)


def cota_del_canal(info, cob, umbral):
    """Cota superior declarada: que pasaria si el canal llegara a las no cubiertas."""
    f1 = {f: info[f]["f1_m6"] for f in info}
    macro_hoy = float(np.mean(list(f1.values())))
    altas = [f for f in f1 if cob.get(f, {}).get("cobertura", 0.0) >= umbral]
    if not altas:
        sys.exit("ABORTA: ninguna familia supera el umbral de cobertura alta.")
    ref = float(np.mean([f1[f] for f in altas]))
    cero = sorted(f for f in f1 if cob.get(f, {}).get("cobertura", 0.0) == 0.0)
    baja = sorted((f for f in f1
                   if 0.0 < cob.get(f, {}).get("cobertura", 0.0) < umbral),
                  key=lambda f: cob[f]["cobertura"])

    def esc(familias):
        nuevo = dict(f1)
        movidas = 0
        for f in familias:
            if nuevo[f] < ref:
                nuevo[f] = ref
                movidas += 1
        return float(np.mean(list(nuevo.values()))), movidas

    return dict(macro_hoy=macro_hoy, referencia=ref, n_altas=len(altas),
                familias_altas=altas, cobertura_cero=cero, cobertura_baja=baja,
                esc_cero=esc(cero), esc_baja=esc(baja), esc_ambas=esc(cero + baja))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=_RES / "margen_notas_149")
    ap.add_argument("--curva", type=Path, default=_RES / "resultados_curva_149",
                    help="carpeta con b1_curva_por_familia.csv")
    ap.add_argument("--m6", type=Path,
                    default=_RES / "resultados_cascada_combinada_149",
                    help="carpeta con m6_diagnostico_nota_por_nota.csv y m6_por_familia.csv")
    ap.add_argument("--techo", type=Path,
                    default=_RES / "resultados_techo_por_familia_149",
                    help="carpeta con techo_por_familia.csv")
    ap.add_argument("--variante", default=VARIANTE_M6)
    args = ap.parse_args()
    OUT = args.salida
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 94)
    print("  DONDE QUEDA MARGEN EN EL FRENTE DE NOTAS: ¿mas notas o otro canal?")
    print("=" * 94)
    info = leer_techo(args.techo)
    cob, variantes = leer_cobertura(args.m6, args.variante)
    faltan = sorted(set(info) - set(cob)) + sorted(set(cob) - set(info))
    if faltan:
        sys.exit(f"ABORTA: las dos fuentes no cubren las mismas familias: {faltan}")
    print(f"  familias: {len(info)} | variante de M.6: {args.variante}")
    print(f"  techo:  {args.techo}")
    print(f"  M.6:    {args.m6}")
    print(f"  curva:  {args.curva}")

    # ---- (1) la tabla por familia
    filas = []
    for fam in sorted(info):
        filas.append(dict(familia=fam, **info[fam], **cob[fam]))
    tab = pd.DataFrame(filas)
    tab.to_csv(OUT / "margen_por_familia.csv", index=False)

    print("\n" + "-" * 94)
    print("  (1) POR FAMILIA, ordenado por cobertura de la regla")
    print("-" * 94)
    print("  %-14s %5s %5s %8s %8s %9s %9s %9s"
          % ("familia", "notas", "plant", "cohes.", "F1 M.6", "cobertura",
             "por nomb.", "texto s/c"))
    for _, r in tab.sort_values("cobertura").iterrows():
        print("  %-14s %5d %5d %8s %8.3f %9.3f %9.3f %9s"
              % (r.familia, r.notas, r.plantillas,
                 "—" if r.cohesion != r.cohesion else "%.3f" % r.cohesion,
                 r.f1_m6, r.cobertura, r.cobertura_por_nombre,
                 "—" if r.acierto_texto_no_cubiertas != r.acierto_texto_no_cubiertas
                 else "%.3f" % r.acierto_texto_no_cubiertas))
    cero = tab[tab.cobertura == 0.0]
    print(f"\n  Familias con cobertura EXACTAMENTE 0 (dependen solo del texto): "
          f"{len(cero)} de {len(tab)}")
    print("    " + ", ".join(sorted(cero.familia)))

    # ---- (2) que predice el F1: cantidad o cohesion
    print("\n" + "-" * 94)
    print("  (2) QUE PREDICE EL F1 POR FAMILIA")
    print("-" * 94)
    conpar = tab[tab.cohesion == tab.cohesion]
    corr = []
    for etiq, x, sub in (("n_plantillas (todas)", "plantillas", tab),
                         ("n_notas (todas)", "notas", tab),
                         ("n_plantillas (solo con cohesion)", "plantillas", conpar),
                         ("cohesion", "cohesion", conpar)):
        for y in ("f1_base", "f1_m6"):
            rho, p = stats.spearmanr(sub[x], sub[y])
            corr.append(dict(x=etiq, y=y, n=len(sub), spearman_rho=rho, p=p))
            print("  %-34s vs %-8s  rho %+.3f (p=%.4f, n=%d)%s"
                  % (etiq, y, rho, p, len(sub), "  <- significativo" if p < 0.05 else ""))
    pd.DataFrame(corr).to_csv(OUT / "correlaciones.csv", index=False)

    # ---- (3) propio vs ajeno
    dd = deltas_por_familia(args.curva, info)
    dd.round(4).to_csv(OUT / "delta_por_familia_propio_ajeno.csv", index=False)
    prop = dd[dd.efecto == "propio"]
    aj = dd[dd.efecto == "ajeno"]

    print("\n" + "-" * 94)
    print("  (3) CUANTO PAGA UNA PLANTILLA MAS -- solo EFECTO PROPIO (el tope le muerde)")
    print("-" * 94)
    for paso in sorted(prop.paso.unique(), key=lambda s: int(s.split("->")[0])):
        s = prop[prop.paso == paso]
        sube = s[s.veredicto == "SUBE"]
        baja = s[s.veredicto == "BAJA"]
        print("  paso %-7s %2d familias | SUBE %2d | BAJA %2d | sin efecto %2d | "
              "Δ mediano %+.4f"
              % (paso, len(s), len(sube), len(baja),
                 len(s) - len(sube) - len(baja), s.delta.median()))
        if len(sube):
            print("      suben: " + ", ".join(
                "%s %+.3f" % (r.familia, r.delta)
                for _, r in sube.sort_values("delta", ascending=False).iterrows()))
        if len(baja):
            print("      BAJAN: " + ", ".join(
                "%s %+.3f" % (r.familia, r.delta)
                for _, r in baja.sort_values("delta").iterrows()))

    conc = prop[prop.cohesion == prop.cohesion]
    rho, p = stats.spearmanr(conc.cohesion, conc.delta)
    print("\n  cohesion vs Δ PROPIO (todos los pasos): rho %+.3f (p=%.4f, n=%d)"
          % (rho, p, len(conc)))
    print("  → la cohesion predice el NIVEL de F1, no el valor marginal de una nota mas."
          if p >= 0.05 else "  → la cohesion tambien predice el valor marginal.")

    print("\n" + "-" * 94)
    print("  (3.bis) EFECTO AJENO: familias a las que el tope NO les muerde")
    print("  (su entrenamiento es identico; cambia que las OTRAS tengan mas dato)")
    print("-" * 94)
    for _, r in aj[aj.veredicto != "sin efecto"].sort_values("delta").iterrows():
        print("  %-14s paso %-7s Δ %+.4f [%+.4f; %+.4f]  %s"
              % (r.familia, r.paso, r.delta, r.ic95_inf, r.ic95_sup, r.veredicto))
    print("  (las que no aparecen quedaron sin efecto medible)")

    # ---- (4) la cota del canal
    c = cota_del_canal(info, cob, UMBRAL_COB_ALTA)
    print("\n" + "-" * 94)
    print("  (4) COTA SUPERIOR DECLARADA DEL CANAL (IOC + nombre)")
    print("-" * 94)
    print("  macro-F1 hoy (M.6): %.4f" % c["macro_hoy"])
    print("  supuesto declarado: las familias no cubiertas alcanzan el F1 medio de las %d"
          % c["n_altas"])
    print("  que hoy tienen cobertura >= %.2f, que es %.4f. NO es una prediccion."
          % (UMBRAL_COB_ALTA, c["referencia"]))
    for etiq, clave, fams in (("las %d de cobertura 0" % len(c["cobertura_cero"]),
                               "esc_cero", c["cobertura_cero"]),
                              ("las %d de cobertura baja" % len(c["cobertura_baja"]),
                               "esc_baja", c["cobertura_baja"]),
                              ("las dos cosas", "esc_ambas",
                               c["cobertura_cero"] + c["cobertura_baja"])):
        macro, mov = c[clave]
        print("    si el canal llegara a %-28s macro-F1 %.4f  (%+.4f, %d familias)"
              % (etiq, macro, macro - c["macro_hoy"], mov))
    filas_c = []
    for fam in c["cobertura_cero"] + c["cobertura_baja"]:
        aporte = max(0.0, c["referencia"] - info[fam]["f1_m6"]) / len(info)
        filas_c.append(dict(familia=fam, cobertura=cob[fam]["cobertura"],
                            f1_m6=info[fam]["f1_m6"], referencia=c["referencia"],
                            aporte_a_la_cota=aporte))
    dc = pd.DataFrame(filas_c).sort_values("aporte_a_la_cota", ascending=False)
    dc.round(4).to_csv(OUT / "cota_del_canal.csv", index=False)
    print("\n  Las que mas aportarian a la cota:")
    for _, r in dc.head(8).iterrows():
        print("    %-14s cobertura %.3f · F1 M.6 %.3f · aporta %+.4f de macro-F1"
              % (r.familia, r.cobertura, r.f1_m6, r.aporte_a_la_cota))

    # ---- manifiesto
    (OUT / "manifiesto_margen.json").write_text(json.dumps(dict(
        fecha=str(date.today()), variante_m6=args.variante,
        curva_clave=list(CURVA_CLAVE), umbral_cobertura_alta=UMBRAL_COB_ALTA,
        fuentes=dict(techo=str(args.techo), m6=str(args.m6), curva=str(args.curva)),
        variantes_en_diagnostico=variantes,
        n_familias=len(info),
        macro_f1_hoy=round(c["macro_hoy"], 4),
        referencia_cobertura_alta=round(c["referencia"], 4),
        familias_cobertura_cero=c["cobertura_cero"],
        cota_solo_cero=round(c["esc_cero"][0] - c["macro_hoy"], 4),
        cota_ambas=round(c["esc_ambas"][0] - c["macro_hoy"], 4),
        aclaracion_cota=("supuesto explicito y falsable, NO prediccion: las familias no "
                         "cubiertas alcanzan el F1 medio de las bien cubiertas"),
        aclaracion_propio_ajeno=("el tope k se aplica a todas las familias a la vez; el "
                                 "Delta es efecto PROPIO solo si n_plantillas >= k+1"),
        pandas=pd.__version__, python=sys.version.split()[0],
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Salidas en: {OUT}")


if __name__ == "__main__":
    main()
