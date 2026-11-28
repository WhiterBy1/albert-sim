"""Anonimiza la exportación de la prueba piloto de Albert ANTES de cualquier uso.

Se ejecuta solo dentro de la UPC. Ni la entrada ni la salida se suben al
repositorio (ver .gitignore): al repositorio solo llegan métricas agregadas.

Entrada (CSV exportado del panel), una fila por mensaje:
    conversacion_id, contacto, fecha_hora, rol, texto[, es_voz]
    rol ∈ {usuario, bot, asesor}
Salida: data/piloto/piloto_anonimizado.csv (solo turnos del usuario) y
        data/piloto/revision_manual.txt (para revisar el 100 % a mano).

Uso:  ALBERT_SAL='<secreto fuera del repo>' python -m src.anonimizar_piloto ruta/export.csv

Replica el filtro de anonimización que Albert aplica antes de llamar a Groq.
Si se dispone del módulo de producción, puede sustituir a `anonimizar_texto`.
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import pandas as pd

from .generar_datos import DATA

SALIDA = DATA / "piloto"

# El orden importa: primero lo más específico.
PATRONES = [
    ("[CORREO]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.I)),
    ("[TEL]", re.compile(r"(?:\+?57[\s-]?)?\b3\d{2}[\s-]?\d{3}[\s-]?\d{4}\b")),
    ("[FECHA]", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("[ORDEN]", re.compile(r"\b(?:orden|factura|solicitud|radicado|ticket|consecutivo)\s*(?:n[o°º.]*|#|:)?\s*[A-Z]*-?\d[\w-]*", re.I)),
    ("[DIRECCION]", re.compile(r"\b(?:calle|cll|carrera|cra|kr|avenida|av|diagonal|dg|transversal|tv)\.?\s*\d+[a-z]?\s*(?:#|no\.?|n°)?\s*\d*[a-z]?\s*-?\s*\d*", re.I)),
    ("[DOC]", re.compile(r"\b(?:\d{1,3}(?:\.\d{3}){2,3}|\d{6,11})\b")),
    ("[TEL]", re.compile(r"\b\d{3}[\s-]?\d{4}\b")),
]
DISPARADORES_NOMBRE = re.compile(
    r"\b((?i:soy|me llamo|mi nombre es|a nombre de|paciente|nombre del paciente|de parte de|mi (?:mam[aá]|pap[aá]|hijo|hija|esposo|esposa) se llama))\s*:?\s+"
    r"((?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})")


def anonimizar_texto(texto: str, nombres_conocidos: set[str] = frozenset()) -> tuple[str, int]:
    """Devuelve el texto anonimizado y el número de reemplazos hechos."""
    n = 0
    t = str(texto)
    t, k = DISPARADORES_NOMBRE.subn(lambda m: f"{m.group(1)} [NOMBRE]", t)
    n += k
    for etiqueta, patron in PATRONES:
        t, k = patron.subn(etiqueta, t)
        n += k
    for nombre in sorted(nombres_conocidos, key=len, reverse=True):   # nombres del panel, si se tienen
        t, k = re.subn(rf"\b{re.escape(nombre)}\b", "[NOMBRE]", t, flags=re.I)
        n += k
    return t, n


def anonimizar(df: pd.DataFrame, sal: str, nombres: set[str] = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # El panel mezcla marcas con y sin fracción de segundo y con zona horaria (-05): ISO8601 + UTC las lee todas.
    df["fecha_hora"] = pd.to_datetime(df.fecha_hora, format="ISO8601", utc=True, errors="coerce")
    df = df.sort_values(["conversacion_id", "fecha_hora"])
    df["es_voz"] = df.get("es_voz", 0)
    df["es_voz"] = df.es_voz.fillna(0).astype(int) | df.texto.fillna("").str.contains(r"audio|nota de voz|\.opus|\.ogg", case=False).astype(int)
    hash_conv = {c: hashlib.sha256(f"{sal}:conv:{c}".encode()).hexdigest()[:16] for c in df.conversacion_id.unique()}

    filas, convs = [], []
    for cid, g in df.groupby("conversacion_id", sort=False):
        t0 = g.fecha_hora.min()
        con_asesor = (g.rol == "asesor").any()
        usuario = g[g.rol == "usuario"].reset_index(drop=True)
        if usuario.empty:
            continue
        # Proxy del turno de escalamiento: último turno del usuario antes del primer mensaje del asesor.
        oraculo = None
        if con_asesor:
            t_asesor = g.loc[g.rol == "asesor", "fecha_hora"].min()
            antes = usuario[usuario.fecha_hora <= t_asesor]
            oraculo = int(len(antes)) if len(antes) else 1
        reemplazos = 0
        for i, r in usuario.iterrows():
            txt, k = ("[NOTA_DE_VOZ]", 0) if r.es_voz else anonimizar_texto(r.texto, nombres)
            reemplazos += k
            filas.append({"conversacion_id": hash_conv[cid], "turno": i + 1, "texto": txt, "es_voz": int(r.es_voz),
                          "segundos_desde_inicio": int((r.fecha_hora - t0).total_seconds()) if pd.notna(r.fecha_hora) else None})
        convs.append({"conversacion_id": hash_conv[cid],
                      "contacto_hash": hashlib.sha256(f"{sal}:{g.contacto.iloc[0]}".encode()).hexdigest(),
                      "requiere_asesor": int(con_asesor), "turno_oraculo": oraculo, "n_turnos": len(usuario),
                      "reemplazos": reemplazos})
    return pd.DataFrame(convs), pd.DataFrame(filas)


COLUMNAS = ["conversacion_id", "contacto", "fecha_hora", "rol", "texto", "es_voz"]


def leer_export(ruta) -> pd.DataFrame:
    """Lee el export; si viene sin encabezado (salida directa de la consulta SQL), asume el orden de COLUMNAS."""
    df = pd.read_csv(ruta)
    if "conversacion_id" in df.columns:
        return df
    df = pd.read_csv(ruta, header=None)
    if df.shape[1] not in (5, 6):
        sys.exit(f"El export no tiene encabezado y trae {df.shape[1]} columnas; se esperaban 5 o 6: {', '.join(COLUMNAS)}.")
    df.columns = COLUMNAS[:df.shape[1]]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_csv")
    ap.add_argument("--nombres", help="archivo opcional con un nombre por línea (nombres registrados en el panel)")
    a = ap.parse_args()
    sal = os.environ.get("ALBERT_SAL")
    if not sal or len(sal) < 16:
        sys.exit("Defina ALBERT_SAL (>= 16 caracteres) fuera del repositorio antes de anonimizar.")
    nombres = set(Path(a.nombres).read_text(encoding="utf-8").split("\n")) - {""} if a.nombres else set()
    convs, turnos = anonimizar(leer_export(a.export_csv), sal, nombres)
    SALIDA.mkdir(parents=True, exist_ok=True)
    convs.to_csv(SALIDA / "piloto_conversaciones.csv", index=False)
    turnos.to_csv(SALIDA / "piloto_anonimizado.csv", index=False)
    with open(SALIDA / "revision_manual.txt", "w", encoding="utf-8") as f:
        for cid, g in turnos.groupby("conversacion_id", sort=False):
            f.write(f"\n=== {cid}  [ ] revisada\n")
            for _, r in g.iterrows():
                f.write(f"  {r.turno:>2}. {r.texto}\n")
    print(f"{len(convs)} conversaciones, {len(turnos)} turnos de usuario, "
          f"{int(convs.reemplazos.sum())} reemplazos. {100 * convs.requiere_asesor.mean():.1f} % terminó con asesor.")
    print(f"Revise a mano: {SALIDA / 'revision_manual.txt'}")


if __name__ == "__main__":
    main()
