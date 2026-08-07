#!/usr/bin/env python3
"""
extractor_notas.py — Capa de extracción multiformato para el clasificador de notas.

Normaliza CUALQUIER tipo de archivo de nota de rescate a texto plano, listo para TF-IDF.
Soporta: .txt y similares, .html/.htm/.hta, .pdf (texto y escaneado vía OCR),
imágenes (.png/.jpg/.jpeg/.bmp/.gif/.tif/.tiff/.webp) vía OCR, .docx y .rtf.

Diseño tolerante a fallos: si una librería/binario no está disponible, degrada con aviso
en vez de romper. Devuelve (texto, metodo) para trazabilidad/citabilidad de cada nota.

Uso:
    from extractor_notas import extraer_texto
    texto, metodo = extraer_texto("ruta/nota.pdf")
"""
from pathlib import Path

# Extensiones tratadas como texto plano (incluye sin extensión y nombres tipo README)
EXT_TEXTO = {"", ".txt", ".text", ".md", ".nfo", ".log", ".message", ".readme",
             ".asc", ".note", ".restore", ".key", ".readme_to_restore"}
EXT_HTML = {".html", ".htm", ".hta", ".xhtml"}
EXT_IMG = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
EXT_PDF = {".pdf"}
EXT_DOCX = {".docx"}
EXT_RTF = {".rtf"}


def _leer_texto_plano(path: Path) -> str:
    """Lee texto detectando la codificación real del archivo.

    Motivación (auditoría 2026-07-27): 14/146 notas de corpus_v2 están en UTF-16
    (DHARMA, GANDCRAB) o cp1252 (SUNCRYPT). Leerlas como UTF-8 con errors="replace"
    no falla: intercala bytes NUL entre letras ('a\\x00l\\x00l\\x00...'), lo que
    produce vectores de palabras vacíos y n-gramas artificiales de caracteres.
    Orden de detección:
      1. BOM UTF-16/UTF-8 (marca explícita de codificación al inicio del archivo)
      2. Heurística de NULs: >20% de bytes 0x00 => UTF-16 sin BOM
         (el lado con más NULs indica si es little-endian o big-endian)
      3. UTF-8 estricto; si falla, cp1252 (superconjunto de latin-1 usado en Windows)
    """
    datos = path.read_bytes()
    if not datos:
        return ""
    # 1. BOM explícito
    if datos[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return datos.decode("utf-16", errors="replace")
    if datos[:3] == b"\xef\xbb\xbf":
        return datos.decode("utf-8-sig", errors="replace")
    # 2. UTF-16 sin BOM: texto ASCII en UTF-16 tiene ~50% de bytes NUL
    if datos.count(0) / len(datos) > 0.20:
        nul_impares = datos[1::2].count(0)
        nul_pares = datos[0::2].count(0)
        codec = "utf-16-le" if nul_impares >= nul_pares else "utf-16-be"
        return datos.decode(codec, errors="replace")
    # 3. UTF-8 estricto con fallback cp1252
    try:
        return datos.decode("utf-8")
    except UnicodeDecodeError:
        return datos.decode("cp1252", errors="replace")


def _limpiar_html_regex(html: str) -> str:
    """Extrae texto visible de HTML sin dependencias externas.

    Fallback para entornos sin BeautifulSoup (p. ej. el cluster del NIDTEC, que no
    tiene acceso a Internet para instalar paquetes). Elimina script/style/comentarios,
    quita etiquetas y decodifica las entidades HTML más comunes.
    """
    import html as _html
    import re
    t = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", html)
    t = re.sub(r"(?s)<!--.*?-->", " ", t)
    t = re.sub(r"(?is)<(meta|link|br|hr)\b[^>]*/?>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t)
    return re.sub(r"[ \t\r\f\v]+", " ", t).strip()


def _leer_html(path: Path) -> str:
    texto = _leer_texto_plano(path)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _limpiar_html_regex(texto)
    soup = BeautifulSoup(texto, "html.parser")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _ocr_imagen(path: Path) -> str:
    import pytesseract
    from PIL import Image
    img = Image.open(path)
    # idiomas: inglés + español (las notas suelen estar en EN; spa por si acaso)
    try:
        return pytesseract.image_to_string(img, lang="eng+spa")
    except Exception:
        return pytesseract.image_to_string(img)  # fallback al idioma por defecto


def _leer_pdf(path: Path):
    """Devuelve (texto, metodo). Intenta texto embebido; si no hay, OCR del PDF."""
    texto = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            texto = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        try:
            from pypdf import PdfReader
            texto = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
        except Exception:
            texto = ""
    if len(texto.strip()) >= 10:
        return texto, "pdf-texto"
    # PDF escaneado -> OCR
    try:
        from pdf2image import convert_from_path
        import pytesseract
        paginas = convert_from_path(str(path), dpi=300)
        texto = "\n".join(pytesseract.image_to_string(pg, lang="eng+spa") for pg in paginas)
        return texto, "pdf-ocr"
    except Exception as e:
        return texto, f"pdf-fallo-ocr({type(e).__name__})"


def _leer_docx(path: Path) -> str:
    import docx
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def _leer_rtf(path: Path) -> str:
    from striprtf.striprtf import rtf_to_text
    return rtf_to_text(_leer_texto_plano(path))


def extraer_texto(path):
    """Extrae texto de una nota de cualquier formato soportado.

    Returns:
        (texto:str, metodo:str)  — metodo indica cómo se extrajo (para trazabilidad).
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in EXT_HTML:
            return _leer_html(path), "html"
        if ext in EXT_PDF:
            return _leer_pdf(path)
        if ext in EXT_IMG:
            return _ocr_imagen(path), "imagen-ocr"
        if ext in EXT_DOCX:
            return _leer_docx(path), "docx"
        if ext in EXT_RTF:
            return _leer_rtf(path), "rtf"
        # texto plano y por defecto (cualquier otra extensión se intenta como texto)
        return _leer_texto_plano(path), "texto" if ext in EXT_TEXTO else "texto-fallback"
    except Exception as e:
        return "", f"error({type(e).__name__}:{e})"


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        t, m = extraer_texto(arg)
        print(f"[{m}] {arg} -> {len(t)} chars")
        print(t[:300])
        print("-" * 50)
