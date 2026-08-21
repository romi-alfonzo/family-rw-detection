#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
construir_embeddings_notas.py -- embedding multilingue de las notas, con TROCEADO.

Se ejecuta como PROCESO APARTE de experimento_embeddings.py a proposito: en Windows,
importar torch DESPUES de scipy/sklearn (MKL/OpenMP ya cargado) puede fallar con
WinError 1114 al cargar c10.dll. Aca torch se importa PRIMERO y este proceso no toca
scipy/sklearn en el camino caliente, asi que el analisis principal queda sin torch.

TROCEADO (resuelve el truncado por max_seq_length):
El modelo tiene max_seq_length=128 tokens; el 86% de las notas del corpus lo superan
(mediana 346 tokens, maximo 12023), asi que un encode directo representaria cada nota
por su comienzo. En cambio se trocea cada nota en ventanas de (max_seq_length-2) tokens
NO solapadas, se embebe cada trozo y se PROMEDIA (mean pooling), y se re-normaliza L2 el
vector resultante. Para una nota que entra en una sola ventana el resultado es identico a
encode(texto, normalize_embeddings=True): el troceado solo cambia las notas largas.

Guarda en <salida>:
  embeddings.npy        matriz float32 [n_notas, dim], L2-normalizada, alineada al orden
                        de cargar_corpus().
  embeddings_meta.json  modelo, versiones, dim, max_seq_length, parametros de troceado y la
                        lista de archivos (para verificar la alineacion fila<->nota).
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
AGREGACION = "chunk_mean_pool_l2"  # etiqueta del metodo (el analisis re-genera si no coincide)


def embeber_troceado(modelo, textos, ventana):
    """Trocea cada nota en ventanas de `ventana` tokens (no solapadas), embebe cada trozo
    (sin normalizar), promedia por nota y re-normaliza L2. Devuelve (emb[n,dim], n_trozos[])."""
    tok = modelo.tokenizer
    trozos, dueno, n_trozos = [], [], []
    for t in textos:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) <= ventana:
            ventanas = [t]  # una sola ventana => identico a encode(t) directo
        else:
            ventanas = [tok.decode(ids[j:j + ventana], skip_special_tokens=True)
                        for j in range(0, len(ids), ventana)]
        n_trozos.append(len(ventanas))
        for w in ventanas:
            trozos.append(w)
            dueno.append(len(n_trozos) - 1)
    emb_trozos = np.asarray(
        modelo.encode(trozos, normalize_embeddings=False, batch_size=64,
                      show_progress_bar=False), dtype=np.float32)
    dueno = np.asarray(dueno)
    dim = emb_trozos.shape[1]
    emb = np.zeros((len(textos), dim), dtype=np.float32)
    for i in range(len(textos)):
        v = emb_trozos[dueno == i].mean(axis=0)
        nrm = np.linalg.norm(v)
        emb[i] = v / nrm if nrm > 0 else v
    return emb, n_trozos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    textos, y, archivos, metodos = cargar_corpus(CORPUS_DIR)
    modelo = SentenceTransformer(MODELO_EMB)
    max_seq = int(modelo.max_seq_length)
    ventana = max_seq - 2  # deja lugar para [CLS]/[SEP]
    print(f"Notas: {len(textos)} | modelo: {MODELO_EMB} | max_seq_length={max_seq} | "
          f"ventana={ventana} tokens")

    emb, n_trozos = embeber_troceado(modelo, textos, ventana)
    n_troceadas = int(sum(1 for c in n_trozos if c > 1))
    print(f"Notas troceadas (>1 ventana): {n_troceadas}/{len(textos)} | "
          f"max trozos por nota: {max(n_trozos)} | dim {emb.shape[1]}")

    np.save(args.salida / "embeddings.npy", emb)
    meta = dict(
        modelo=MODELO_EMB,
        sentence_transformers=sentence_transformers.__version__,
        torch=torch.__version__,
        transformers=transformers.__version__,
        dim=int(emb.shape[1]),
        max_seq_length=max_seq,
        normalize_embeddings=True,
        aggregation=AGREGACION,
        window_tokens=int(ventana),
        stride_tokens=int(ventana),   # ventanas NO solapadas
        n_notas=int(len(textos)),
        n_troceadas=n_troceadas,
        max_chunks=int(max(n_trozos)),
        corpus=str(CORPUS_DIR),
        archivos=archivos,  # orden exacto de cargar_corpus, para verificar alineacion
    )
    (args.salida / "embeddings_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {args.salida / 'embeddings.npy'} {emb.shape} | agg {AGREGACION}")


if __name__ == "__main__":
    main()
