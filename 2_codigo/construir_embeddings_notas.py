#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construir_embeddings_notas.py -- calcula el embedding multilingue de las notas y lo guarda.

Se ejecuta como PROCESO APARTE de experimento_embeddings.py a proposito: en Windows,
importar torch DESPUES de scipy/sklearn (MKL/OpenMP ya cargado) puede fallar con
WinError 1114 al cargar c10.dll. Aca torch se importa PRIMERO y este proceso no toca
scipy/sklearn en el camino caliente, asi que el analisis principal queda sin torch.

Guarda en <salida>:
  embeddings.npy        matriz float32 [n_notas, dim], L2-normalizada, alineada al orden
                        de cargar_corpus().
  embeddings_meta.json  modelo, versiones, dim, max_seq_length y la lista de archivos (para
                        que el analisis verifique la alineacion fila<->nota).
"""
import torch  # PRIMERO: carga las DLL de torch antes que MKL (evita WinError 1114 en Windows)
from sentence_transformers import SentenceTransformer
import sentence_transformers
import transformers

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI))
# cargar_corpus vive en clasificador_notas_v2 (importa sklearn), pero torch YA se cargo arriba.
from clasificador_notas_v2 import CORPUS_DIR, cargar_corpus

MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    print(f"Notas cargadas: {len(textos)} | modelo: {MODELO_EMB}")

    modelo = SentenceTransformer(MODELO_EMB)
    emb = modelo.encode(list(textos), normalize_embeddings=True, batch_size=32,
                        show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(args.salida / "embeddings.npy", emb)

    meta = dict(
        modelo=MODELO_EMB,
        sentence_transformers=sentence_transformers.__version__,
        torch=torch.__version__,
        transformers=transformers.__version__,
        dim=int(emb.shape[1]),
        max_seq_length=int(getattr(modelo, "max_seq_length", -1)),
        normalize_embeddings=True,
        n_notas=int(len(textos)),
        corpus=str(CORPUS_DIR),
        archivos=archivos,  # orden exacto de cargar_corpus, para verificar alineacion
    )
    (args.salida / "embeddings_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {args.salida / 'embeddings.npy'} {emb.shape} | "
          f"dim {meta['dim']} | max_seq_length {meta['max_seq_length']}")


if __name__ == "__main__":
    main()
