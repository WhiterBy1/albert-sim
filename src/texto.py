"""Utilidades de texto compartidas: normalización y ruido de escritura."""
import re
import unicodedata

_TILDES = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin signos: así compara Albert las palabras clave."""
    t = unicodedata.normalize("NFC", texto or "").translate(_TILDES).lower()
    t = re.sub(r"[^a-z0-9\[\] ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(texto: str) -> set:
    return set(normalizar(texto).split())


_ABREV = {"que": "q", "porque": "xq", "por": "x", "para": "pa", "también": "tmb",
          "favor": "fa", "mensaje": "msj", "cuando": "cdo"}


def ensuciar(texto: str, rng, tasa: float) -> str:
    """Simula mala escritura: quita tildes, abrevia, omite y duplica letras.

    `tasa` es la probabilidad por palabra de sufrir una alteración.
    """
    salida = []
    for palabra in texto.split():
        base = palabra.lower()
        if rng.random() < tasa:
            op = rng.integers(0, 4)
            if op == 0 and base in _ABREV:
                palabra = _ABREV[base]
            elif op == 1 and len(palabra) > 4:
                i = int(rng.integers(1, len(palabra) - 1))
                palabra = palabra[:i] + palabra[i + 1:]          # omite una letra
            elif op == 2 and len(palabra) > 3:
                i = int(rng.integers(1, len(palabra) - 1))
                palabra = palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]  # transpone
            else:
                palabra = palabra.translate(_TILDES)
        salida.append(palabra)
    t = " ".join(salida)
    if rng.random() < tasa:
        t = t.lower().rstrip(".?!")
    return t
