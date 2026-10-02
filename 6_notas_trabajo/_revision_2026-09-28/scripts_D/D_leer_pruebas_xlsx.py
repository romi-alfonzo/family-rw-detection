#!/usr/bin/env python3
"""D_leer_pruebas_xlsx.py — Vuelca todas las hojas de Pruebas.xlsx (evaluación de herramientas) a
salidas_D/D_pruebas_xlsx.txt, y recuenta aciertos por hoja. Solo lectura."""
from pathlib import Path

import openpyxl

RUTA = Path(r"C:\Users\Romina\Tesis\7_compartido_carlos\Tesis Carlos y Romina\Pruebas.xlsx")
SAL = Path(r"C:\Users\Romina\Tesis\6_notas_trabajo\_revision_2026-09-28\salidas_D")
SAL.mkdir(parents=True, exist_ok=True)

wb = openpyxl.load_workbook(RUTA, data_only=True, read_only=True)
with open(SAL / "D_pruebas_xlsx.txt", "w", encoding="utf-8") as fh:
    fh.write(f"### ORIGEN: {RUTA}\n### HOJAS: {wb.sheetnames}\n")
    for nombre in wb.sheetnames:
        ws = wb[nombre]
        fh.write(f"\n\n===== HOJA: {nombre} (dims {ws.dimensions}) =====\n")
        for i, fila in enumerate(ws.iter_rows(values_only=True), start=1):
            if all(c is None for c in fila):
                continue
            celdas = ["" if c is None else str(c).replace("\n", " / ") for c in fila]
            fh.write(f"{i:4d} | " + " | ".join(celdas) + "\n")
print("hojas:", wb.sheetnames)
print("escrito en", SAL / "D_pruebas_xlsx.txt")
