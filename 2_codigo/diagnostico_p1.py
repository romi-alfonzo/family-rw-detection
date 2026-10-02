#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diagnostico_p1.py -- donde falla la cascada cuando la plantilla «ya se conoce».

Pregunta de Romina (2026-10-01): si bajo P1 la plantilla ya esta catalogada, por que el sistema
no acierta casi siempre (0,8866). Este script reproduce el bucle de abstencion_notas.py bajo P1
y registra, nota por nota y semilla por semilla, tres cosas que la corrida no guardaba:
  - si alguna HERMANA de su plantilla quedo en entrenamiento (si no, P1 es tan exigente como P2
    para esa nota: con 99 plantillas para 149 notas, decenas de plantillas tienen UNA sola nota);
  - que capa decidio (regla o texto) y si acerto;
  - la familia.
No cambia ningun calculo; solo mira. Salida: 4_resultados/resultados_diagnostico_P1/.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstencion_notas as ab  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

N_SEM = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = ab.RAIZ / "4_resultados" / "resultados_diagnostico_P1"
OUT.mkdir(parents=True, exist_ok=True)

textos, y, archivos, _ = ab.cargar_corpus(ab.CORPUS_DIR)
grupos, _ = ab.agrupar_neardups(textos, ab.UMBRAL_NEARDUP)
textos_arr = np.array(textos, dtype=object)
y = np.asarray(y)
grupos = np.asarray(grupos)
n = len(textos)
iocs = [set(ab.extraer_marcadores(t)) for t in textos]
nom_aud = ab.cargar_nombres()
nombres_nota = [nom_aud.get((f, Path(a).name)) for f, a in zip(y, archivos)]

tam_plantilla = Counter(grupos)
singletons = sum(1 for i in range(n) if tam_plantilla[grupos[i]] == 1)
print(f"Notas: {n} | Plantillas: {len(set(grupos))} | notas cuya plantilla tiene UNA sola nota: {singletons}")

# registro[(hermana_en_train, capa)] -> [aciertos, decisiones]
reg = defaultdict(lambda: [0, 0])
err_fam = Counter()
dec_fam = Counter()
for s in range(N_SEM):
    cv = StratifiedKFold(n_splits=ab.N_FOLDS, shuffle=True, random_state=s)
    for tr, te in cv.split(textos_arr, y):
        vec = ab.vectorizador("combinado")
        Xtr = vec.fit_transform(textos_arr[tr])
        Xte = vec.transform(textos_arr[te])
        clf = ab.obtener_modelos(s)["LinearSVC"]
        clf.fit(Xtr, y[tr])
        top1 = clf.classes_[np.argmax(clf.decision_function(Xte), axis=1)]
        d = ab.dicc_privados(tr, iocs, nombres_nota, y)
        grupos_tr = set(grupos[tr])
        for k, i in enumerate(te):
            claves = set(iocs[i])
            if nombres_nota[i]:
                claves.add(("[NOMBRE]", nombres_nota[i]))
            fams = set()
            for c in claves:
                if c in d:
                    fams |= d[c]
            if len(fams) == 1:
                capa, pred = "regla", next(iter(fams))
            else:
                capa, pred = "texto", top1[k]
            hermana = grupos[i] in grupos_tr
            ok = pred == y[i]
            reg[(hermana, capa)][0] += ok
            reg[(hermana, capa)][1] += 1
            dec_fam[y[i]] += 1
            if not ok:
                err_fam[y[i]] += 1

tot_dec = sum(v[1] for v in reg.values())
tot_err = sum(v[1] - v[0] for v in reg.values())
print(f"\nSemillas: {N_SEM} | decisiones: {tot_dec} | errores: {tot_err} ({100*tot_err/tot_dec:.1f} %)\n")
print(f"{'hermana en train':<18}{'capa':<8}{'decisiones':>12}{'acierto':>10}{'% del error':>13}")
for (h, capa), (ac, dec) in sorted(reg.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
    print(f"{('SI' if h else 'NO'):<18}{capa:<8}{dec:>12}{ac/dec:>10.4f}{100*(dec-ac)/tot_err:>12.1f} %")

print("\nFamilias que mas error aportan (errores / decisiones, acierto):")
for fam, e in err_fam.most_common(8):
    print(f"  {fam:<14}{e:>6} / {dec_fam[fam]:<6} {1 - e/dec_fam[fam]:.3f}")

with open(OUT / "diagnostico_P1.csv", "w", encoding="utf-8") as f:
    f.write("hermana_en_train,capa,decisiones,aciertos\n")
    for (h, capa), (ac, dec) in reg.items():
        f.write(f"{int(h)},{capa},{dec},{ac}\n")
print(f"\nSalida: {OUT}")
