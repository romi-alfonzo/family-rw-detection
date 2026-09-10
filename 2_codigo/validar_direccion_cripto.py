"""Valida direcciones de criptomoneda por su CHECKSUM, no por lectura.

PARA QUE SIRVE EN ESTA TESIS
  Varias notas del corpus entraron por OCR de capturas de pantalla, y la direccion de pago
  es el IOC mas valioso de una nota (es el marcador privado que usa la regla de M.6). Un
  caracter mal leido convierte un dato verificable en un dato falso, y en un monospace de
  terminal las confusiones l/1, O/0, s/3, M/H son justamente las mas probables.

  Estas direcciones llevan checksum, asi que se pueden verificar solas:
    - Bitcoin (P2PKH/P2SH, empieza con 1 o 3): Base58Check, 4 bytes de doble SHA-256.
    - Monero (empieza con 4 o 8): base58 de Monero por bloques + 4 bytes de Keccak-256.
  Si el checksum cierra, la transcripcion es correcta; si no cierra, hay un error de
  lectura y el script prueba las confusiones tipicas para ubicarlo.

  Keccak-256 va implementado aca porque no hay libreria en el entorno (hashlib.sha3_256 NO
  sirve: usa el relleno de NIST, 0x06, y Monero usa el Keccak original, 0x01). Se
  autoverifica contra vectores conocidos ANTES de dictaminar: si el autotest falla, aborta
  en vez de dar un veredicto que no vale.

Uso:
    python validar_direccion_cripto.py <direccion> [otra ...]
    python validar_direccion_cripto.py --archivo nota.txt      (busca y valida las que haya)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ALFABETO = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
# Monero codifica en bloques: n bytes -> cuantos caracteres ocupa
BLOQUE_A_CHARS = [0, 2, 3, 5, 6, 7, 9, 10, 11]

# ============================================================
# Keccak-256 (Keccak original, relleno 0x01 — NO es SHA3-256)
# ============================================================
_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
       0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
       0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
       0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
       0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
       0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
       0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
       0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
        [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M = (1 << 64) - 1


def _rotl(x, n):
    n %= 64
    return ((x << n) | (x >> (64 - n))) & _M


def _keccak_f(a):
    for rnd in range(24):
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl(a[x][y], _ROT[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y] & _M) & b[(x + 2) % 5][y])
        a[0][0] ^= _RC[rnd]
    return a


def keccak256(datos: bytes) -> bytes:
    tasa = 136                                  # 1088 bits
    m = bytearray(datos)
    m.append(0x01)                              # relleno del Keccak ORIGINAL
    while len(m) % tasa != 0:
        m.append(0x00)
    m[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(m), tasa):
        bloque = m[off:off + tasa]
        for i in range(tasa // 8):
            palabra = int.from_bytes(bloque[i * 8:(i + 1) * 8], "little")
            a[i % 5][i // 5] ^= palabra
        a = _keccak_f(a)
    salida = b""
    for i in range(4):                          # 32 bytes
        salida += a[i % 5][i // 5].to_bytes(8, "little")
    return salida[:32]


def autotest_keccak():
    """Sin esto el veredicto no vale nada. Vectores publicos de Keccak-256."""
    casos = [
        (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        (b"abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
    ]
    for datos, esperado in casos:
        obtenido = keccak256(datos).hex()
        if obtenido != esperado:
            sys.exit(f"ABORTA: el autotest de Keccak-256 fallo con {datos!r}.\n"
                     f"  esperado {esperado}\n  obtenido {obtenido}\n"
                     f"No se emite ningun veredicto con una implementacion sin verificar.")
    return True


# ============================================================
# Base58
# ============================================================
def _b58_a_int(s):
    n = 0
    for c in s:
        if c not in ALFABETO:
            raise ValueError(f"caracter fuera del alfabeto Base58: {c!r}")
        n = n * 58 + ALFABETO.index(c)
    return n


def valida_btc(dir_):
    try:
        n = _b58_a_int(dir_)
    except ValueError as e:
        return False, str(e)
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    b = b"\x00" * (len(dir_) - len(dir_.lstrip("1"))) + b
    if len(b) != 25:
        return False, f"decodifica a {len(b)} bytes, se esperaban 25"
    calc = hashlib.sha256(hashlib.sha256(b[:21]).digest()).digest()[:4]
    return b[21:] == calc, f"checksum leido {b[21:].hex()} vs calculado {calc.hex()}"


def valida_xmr(dir_):
    datos = bytearray()
    for i in range(0, len(dir_), 11):
        bloque = dir_[i:i + 11]
        if len(bloque) not in BLOQUE_A_CHARS:
            return False, f"bloque de {len(bloque)} caracteres: largo imposible en Monero"
        n_bytes = BLOQUE_A_CHARS.index(len(bloque))
        try:
            v = _b58_a_int(bloque)
        except ValueError as e:
            return False, str(e)
        if v >= 1 << (8 * n_bytes):
            return False, f"el bloque {bloque!r} desborda sus {n_bytes} bytes"
        datos += v.to_bytes(n_bytes, "big")
    if len(datos) < 5:
        return False, "demasiado corta"
    cuerpo, chk = bytes(datos[:-4]), bytes(datos[-4:])
    calc = keccak256(cuerpo)[:4]
    detalle = (f"{len(datos)} bytes | red 0x{datos[0]:02x} | "
               f"checksum leido {chk.hex()} vs calculado {calc.hex()}")
    return chk == calc, detalle


CONFUSIONES = {"1": "lI", "l": "1I", "I": "1l", "0": "O", "O": "0", "5": "S", "S": "5",
               "3": "s8", "s": "35", "8": "3B", "B": "8", "6": "Gb", "G": "6",
               "M": "HN", "H": "M", "N": "M", "2": "Zz", "Z": "2", "z": "2",
               "9": "gq", "g": "9q", "q": "9g", "u": "vn", "v": "u", "w": "vv"}


def diagnosticar(dir_, validador):
    """Si no cierra, prueba las confusiones tipicas de OCR de a una."""
    hallados = []
    for i, c in enumerate(dir_):
        for alt in CONFUSIONES.get(c, ""):
            cand = dir_[:i] + alt + dir_[i + 1:]
            ok, _ = validador(cand)
            if ok:
                hallados.append((i, c, alt, cand))
    return hallados


def clasificar(dir_):
    if dir_[:1] in ("1", "3") and 26 <= len(dir_) <= 35:
        return "BTC", valida_btc
    if dir_[:1] in ("4", "8") and len(dir_) in (95, 106):
        return "XMR", valida_xmr
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direcciones", nargs="*")
    ap.add_argument("--archivo", type=Path, action="append", default=[],
                    help="archivo de texto del que extraer direcciones candidatas")
    args = ap.parse_args()

    autotest_keccak()
    print("autotest de Keccak-256: OK (2 vectores publicos)\n")

    cands = list(args.direcciones)
    for f in args.archivo:
        txt = f.read_text(encoding="utf-8", errors="replace")
        # Monero: 95 o 106 chars; Bitcoin: 26-35. Se admite el corte de linea en medio.
        plano = re.sub(r"[\r\n]+", "", txt)
        cands += re.findall(r"\b[48][1-9A-HJ-NP-Za-km-z]{94,105}", plano)
        cands += re.findall(r"\b[13][1-9A-HJ-NP-Za-km-z]{25,34}\b", txt)
        print(f"{f.name}: {len(cands)} candidata(s) acumulada(s)")

    if not cands:
        sys.exit("No hay direcciones para validar.")

    for d in cands:
        tipo, validador = clasificar(d)
        print("-" * 78)
        print(f"{d}")
        print(f"  largo {len(d)} | tipo detectado: {tipo or 'DESCONOCIDO'}")
        if not validador:
            print("  no se puede validar: no coincide con ningun formato conocido")
            continue
        ok, detalle = validador(d)
        print(f"  {detalle}")
        print(f"  CHECKSUM: {'VALIDO' if ok else 'NO VALIDO'}")
        if ok:
            print("  => transcripcion correcta (un solo caracter mal leido lo rompe)")
        else:
            alt = diagnosticar(d, validador)
            if alt:
                print("  => hay una variante de UN caracter que SI valida:")
                for i, c, a, cand in alt:
                    print(f"     posicion {i}: '{c}' deberia ser '{a}'")
                    print(f"     {cand}")
            else:
                print("  => no cierra, y ninguna confusion de un solo caracter lo arregla: "
                      "hay mas de un error, o la direccion es inventada")


if __name__ == "__main__":
    main()
