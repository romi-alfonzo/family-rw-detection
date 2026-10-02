#!/usr/bin/env python3
"""D_extraer_pdfs.py — Extrae el texto de cada PDF de 5_bibliografia/ a salidas_D/D_pdf_<nombre>.txt
con marcadores de página, para poder buscar cifras y citar página. Solo lectura de los PDF."""
import sys
from pathlib import Path

from pypdf import PdfReader

RAIZ = Path(r"C:\Users\Romina\Tesis")
BIB = RAIZ / "5_bibliografia"
SAL = RAIZ / "6_notas_trabajo" / "_revision_2026-09-28" / "salidas_D"
SAL.mkdir(parents=True, exist_ok=True)

resumen = []
for pdf in sorted(BIB.rglob("*.pdf")):
    nombre = pdf.stem.replace(" ", "_").replace("(", "").replace(")", "")
    destino = SAL / f"D_pdf_{nombre}.txt"
    try:
        lector = PdfReader(str(pdf))
        paginas = len(lector.pages)
        with open(destino, "w", encoding="utf-8") as fh:
            fh.write(f"### ORIGEN: {pdf}\n### PAGINAS: {paginas}\n")
            for i, pag in enumerate(lector.pages, start=1):
                try:
                    txt = pag.extract_text() or ""
                except Exception as e:  # noqa
                    txt = f"[error extrayendo: {e}]"
                fh.write(f"\n\n===== PAGINA {i} =====\n{txt}")
        primera = (lector.pages[0].extract_text() or "")[:200].replace("\n", " ")
        resumen.append(f"{pdf.relative_to(BIB)} | {paginas} pág. | {primera}")
    except Exception as e:  # noqa
        resumen.append(f"{pdf.relative_to(BIB)} | ERROR: {e}")

with open(SAL / "D_pdf_indice.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(resumen))
print("\n".join(resumen))
