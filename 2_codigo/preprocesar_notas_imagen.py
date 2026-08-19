#!/usr/bin/env python3
"""
preprocesar_notas_imagen.py — De captura de nota de rescate a candidato de texto.

Flujo pensado: Romina descarga las imágenes a mano (capturas de notas en pcrisk,
Malwarebytes, etc.), las deja en una carpeta, corre este script, y obtiene el texto
por OCR + la metadata de procedencia, listo para pasarle verificar_nota_nueva.py.
NO copia nada al corpus ni toca el manifiesto: eso queda manual, después de verificar
que el texto sea nuevo.

──────────────────────────────────────────────────────────────────────────────
SOBRE «DE QUÉ TIPO DE ARCHIVO ES» — leer, es la parte que se malinterpreta fácil:

  La extensión original de una nota NO se puede deducir de los píxeles de una
  captura. Una imagen de una nota renderizada no distingue .txt de .html de una
  ventana de aplicación. Ese dato es DOCUMENTAL: sale del nombre de archivo que
  reporta el vendor (p. ej. YOUR_FILES_ARE_ENCRYPTED.HTML), de la firma del
  antivirus (Ransom:HTML/Chicrypt.A ⇒ HTML) o del comportamiento descrito.

  Por eso el script separa DOS cosas y no las mezcla:
   1. `extension_original`  → la ponés vos en el CSV de procedencia, desde la doc.
      Es el dato que va al manifiesto. Es documentado, no observado.
   2. `pista_visual`        → heurística débil que mira la imagen y sugiere si es
      una VENTANA (bordes/barra de título/mucho color de UI), un DOCUMENTO de texto
      (fondo liso, texto monoespaciado) o HTML con estilos. Sirve solo como CONTROL
      de consistencia con (1). Nunca reemplaza a la doc.

  Y NO se «recrea» el .html: guardar el texto como .html envolviéndolo en marcado
  inventado sería fabricar el artefacto. Se guarda .txt (es texto) y se registra la
  extensión original aparte. Esa es la política del corpus.
──────────────────────────────────────────────────────────────────────────────

Requisitos para el OCR: tesseract instalado + pytesseract + Pillow.
  winget install --id UB-Mannheim.TesseractOCR
  pip install pytesseract pillow
Para notas no inglesas conviene tener los packs de idioma de tesseract (deu, fra,
spa). Si faltan, el script cae a inglés y lo avisa.
Sin tesseract, el script IGUAL corre: hace la pista visual y arma el CSV, y deja el
OCR pendiente (útil para preparar la metadata antes de instalar nada).

Uso:
  python preprocesar_notas_imagen.py <carpeta_con_imagenes>

La carpeta debe tener un `procedencia.csv` (se crea una plantilla si no existe) con:
  imagen, familia, extension_original, fuente, url, idioma
`idioma` es opcional (eng/deu/fra/spa o combinaciones «eng+deu»); por defecto eng.

Salida en la misma carpeta:
  <basename>.ocr.txt        el texto extraído de cada imagen
  candidatos_ocr.csv        tabla con texto + metadata, para revisar y verificar
"""

import csv
import sys
from pathlib import Path

from PIL import Image

# La consola de Windows es cp1252; forzamos UTF-8 para no romper con acentos/flechas.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXT_IMG = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ────────────────────────────────────────────────────────────────────────────
# OCR con preprocesamiento (upscale ×3 + binarizado; es lo que funcionó para la
# pantalla MBR de NotPetya). Degrada con aviso si falta tesseract/pytesseract.
# ────────────────────────────────────────────────────────────────────────────
def _localizar_tesseract(pytesseract):
    """El instalador de Windows no siempre deja tesseract en el PATH. Si no está,
    lo buscamos en las rutas estándar y se lo indicamos a pytesseract."""
    import os
    import shutil
    if shutil.which("tesseract"):
        return  # ya visible en el PATH
    candidatos = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidatos:
        if Path(c).is_file():
            pytesseract.pytesseract.tesseract_cmd = c
            return


def ocr_imagen(path: Path, idioma: str) -> tuple[str, str]:
    """Devuelve (texto, metodo). metodo = 'ocr' | 'ocr-eng-fallback' | 'sin-ocr(...)'."""
    try:
        import pytesseract
    except ImportError:
        return "", "sin-ocr(falta pytesseract)"
    _localizar_tesseract(pytesseract)

    try:
        img = Image.open(path).convert("L")            # escala de grises
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)  # upscale ×3
        img = img.point(lambda p: 0 if p < 140 else 255)  # binarizado simple
    except Exception as e:
        return "", f"sin-ocr(imagen ilegible: {type(e).__name__})"

    try:
        return pytesseract.image_to_string(img, lang=idioma), "ocr"
    except pytesseract.TesseractNotFoundError:
        return "", "sin-ocr(falta el binario tesseract)"
    except Exception:
        # idioma no disponible u otro error → reintento en inglés
        try:
            return pytesseract.image_to_string(img), "ocr-eng-fallback"
        except Exception as e:
            return "", f"sin-ocr({type(e).__name__})"


# ────────────────────────────────────────────────────────────────────────────
# Pista visual: ventana / documento / html. HEURÍSTICA DÉBIL, solo control.
# ────────────────────────────────────────────────────────────────────────────
def pista_visual(path: Path) -> str:
    """Sugiere el tipo de render a partir de la imagen. NO es la extensión real:
    es un control de consistencia con lo documentado. Mira dos señales baratas:
    - fracción de píxeles «de color» (ni casi-blanco ni casi-negro ni gris): mucha
      señal de UI/HTML con estilos; casi nada ⇒ texto plano sobre fondo liso.
    - banda superior homogénea distinta del cuerpo: típica barra de título ⇒ ventana.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return f"indeterminada(imagen ilegible: {type(e).__name__})"

    img_p = img.resize((160, 120))
    px = list(img_p.getdata())
    n = len(px)

    def coloreado(p):
        r, g, b = p
        return (max(r, g, b) - min(r, g, b)) > 40  # canal separado ⇒ color, no gris

    frac_color = sum(coloreado(p) for p in px) / n

    # banda superior (10%) vs cuerpo: ¿color medio muy distinto? ⇒ barra de título
    w, h = img_p.size
    franja = [px[y * w + x] for y in range(max(1, h // 10)) for x in range(w)]
    cuerpo = [px[y * w + x] for y in range(h // 5, h) for x in range(w)]
    prom = lambda s: tuple(sum(c) / len(s) for c in zip(*s))
    pf, pc = prom(franja), prom(cuerpo)
    dif_banda = sum(abs(a - b) for a, b in zip(pf, pc))

    if dif_banda > 60 and frac_color > 0.05:
        return f"ventana-app? (barra sup. dif={dif_banda:.0f}, color={frac_color:.0%})"
    if frac_color > 0.15:
        return f"html-con-estilos? (color={frac_color:.0%})"
    return f"documento-texto? (color={frac_color:.0%})"


# ────────────────────────────────────────────────────────────────────────────
def plantilla_procedencia(carpeta: Path, imagenes: list[Path]) -> Path:
    """Crea procedencia.csv con una fila por imagen para que Romina la complete."""
    p = carpeta / "procedencia.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "familia", "extension_original", "fuente", "url", "idioma"])
        for img in imagenes:
            w.writerow([img.name, "", "", "", "", "eng"])
    return p


def cargar_procedencia(carpeta: Path) -> dict:
    p = carpeta / "procedencia.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return {fila["imagen"]: fila for fila in csv.DictReader(f)}


def main():
    if len(sys.argv) != 2:
        sys.exit("Uso: python preprocesar_notas_imagen.py <carpeta_con_imagenes>")
    carpeta = Path(sys.argv[1])
    if not carpeta.is_dir():
        sys.exit(f"No es una carpeta: {carpeta}")

    imagenes = sorted(p for p in carpeta.iterdir()
                      if p.is_file() and p.suffix.lower() in EXT_IMG)
    if not imagenes:
        sys.exit(f"No hay imágenes en {carpeta}")

    proc = cargar_procedencia(carpeta)
    if not proc:
        p = plantilla_procedencia(carpeta, imagenes)
        print(f"Creé la plantilla {p.name}: complétala (extension_original, fuente, url, "
              f"idioma) y volvé a correr. Igual proceso el OCR y la pista visual ahora.")

    filas = []
    for img in imagenes:
        meta = proc.get(img.name, {})
        idioma = (meta.get("idioma") or "eng").strip() or "eng"
        texto, metodo = ocr_imagen(img, idioma)
        pista = pista_visual(img)

        salida = carpeta / (img.stem + ".ocr.txt")
        if texto.strip():
            salida.write_text(texto, encoding="utf-8")

        filas.append(dict(
            imagen=img.name,
            familia=meta.get("familia", ""),
            extension_original_documentada=meta.get("extension_original", ""),
            pista_visual=pista,
            metodo_ocr=metodo,
            n_chars=len(texto.strip()),
            archivo_texto=salida.name if texto.strip() else "",
            fuente=meta.get("fuente", ""),
            url=meta.get("url", ""),
        ))
        print(f"  {img.name:<34} {metodo:<28} {len(texto.strip()):>5} chars   {pista}")

    out = carpeta / "candidatos_ocr.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"\n{len(filas)} imagen(es) procesadas → {out.name}")
    print("Siguiente paso: revisar los .ocr.txt (el OCR puede errar), y para cada uno\n"
          "  python verificar_nota_nueva.py <ruta>.ocr.txt\n"
          "Solo si da «TEXTO NUEVO» se copia a corpus_v2/<FAMILIA>/ y se agrega al\n"
          "manifiesto con la extension_original DOCUMENTADA (la columna, no la pista visual).")


if __name__ == "__main__":
    main()
