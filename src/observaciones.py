"""Sensor del agente: convierte cada turno del usuario en una observación discreta.

Reproduce la lógica de Albert: palabras clave por frecuencia para decidir si hay
"atajo" (ruta por reglas). Sobre eso agrega las señales que usan las reglas de
escalamiento y el HMM. Todo trabaja sobre texto ya anonimizado.
"""
from .texto import normalizar, tokens

# Palabras clave de las rutas por reglas (atajos) de Albert. Si el turno contiene
# alguna, el bot responde por regla; si no, cae al LLM ("sin atajo").
RUTAS = {
    "horarios": ["horario", "horarios", "hora", "abren", "cierran", "atienden", "domingo", "sabado", "festivo"],
    "sedes": ["sede", "sedes", "direccion", "ubicacion", "donde", "queda", "sucursal"],
    "pagos": ["pago", "pagar", "precio", "valor", "cuesta", "transferencia", "nequi", "tarjeta"],
    "estado_estudio": ["resultado", "resultados", "examen", "examenes", "estudio", "listo", "muestra"],
    "requisitos": ["orden", "documento", "llevar", "parqueadero", "demora", "cedula", "autorizacion"],
    "saludo_cierre": ["gracias", "hola", "buenos dias", "buenas tardes", "amable", "perfecto"],
    "preparacion": ["ayuno", "ayunas", "preparacion", "prepararme", "orina", "tomar agua"],
}

PIDE_ASESOR = ["asesor", "asesora", "persona", "humano", "operador", "alguien real", "funcionario",
               "comunicarme con alguien", "hablar con alguien", "agente"]
NEGATIVO = ["no sirve", "pesimo", "pesima", "molesto", "molesta", "urgente", "llevo dias", "nadie responde",
            "ya les dije", "falta de seriedad", "cansado", "cansada", "absurdo", "inaceptable", "queja",
            "reclamo", "no entiende", "no me ayuda", "otra vez", "de nuevo lo mismo"]
CIERRE = ["gracias", "muchas gracias", "listo gracias", "perfecto", "muy amable", "ok gracias", "vale gracias"]

# Orden de prioridad: si un turno dispara varias señales se conserva la primera.
SIMBOLOS = ["PIDE_ASESOR", "NEGATIVO", "REPITE", "VOZ", "SIN_ATAJO", "CIERRE", "ATAJO"]
IDX = {s: i for i, s in enumerate(SIMBOLOS)}


def _contiene(texto_norm: str, lista) -> bool:
    relleno = f" {texto_norm} "
    return any(f" {normalizar(p)} " in relleno for p in lista)


def ruta_atajo(texto: str):
    """Ruta por reglas que Albert tomaría, o None si no hay atajo (va al LLM)."""
    t = normalizar(texto)
    conteo = {r: sum(f" {p} " in f" {t} " for p in pals) for r, pals in RUTAS.items()}
    mejor = max(conteo, key=conteo.get)
    return mejor if conteo[mejor] > 0 else None


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def observar(texto: str, es_voz: bool, texto_previo: str | None) -> dict:
    """Devuelve las señales binarias del turno y el símbolo discreto (por prioridad)."""
    t = normalizar(texto)
    senales = {
        "PIDE_ASESOR": (not es_voz) and _contiene(t, PIDE_ASESOR),
        "NEGATIVO": (not es_voz) and _contiene(t, NEGATIVO),
        "REPITE": (not es_voz) and texto_previo is not None and jaccard(tokens(texto), tokens(texto_previo)) >= 0.5,
        "VOZ": bool(es_voz),
        "SIN_ATAJO": es_voz or ruta_atajo(texto) is None,
        "CIERRE": (not es_voz) and _contiene(t, CIERRE) and len(t.split()) <= 6,
    }
    senales["ATAJO"] = not senales["SIN_ATAJO"]
    simbolo = next(s for s in SIMBOLOS if senales[s])
    if simbolo == "SIN_ATAJO" and senales["CIERRE"]:
        simbolo = "CIERRE"   # un "gracias" corto no es un fallo del bot
    return {**{f"s_{k.lower()}": int(v) for k, v in senales.items()}, "obs": simbolo}
