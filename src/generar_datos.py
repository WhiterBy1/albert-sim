"""Generador de conversaciones sintéticas para Albert-sim.

No usa un HMM para generar (eso volvería circular la evaluación). Cada conversación
nace de un *caso* (intención + complejidad) y de una variable continua de
frustración que crece con los fallos del bot. El caso "requiere asesor" si es
complejo o si la frustración supera un umbral. El turno en que la necesidad se
hace evidente queda guardado como `turno_oraculo` para medir el retraso.

Hay dos variantes con plantillas y parámetros distintos:
  G1 -> entrenamiento y validación
  G2 -> prueba independiente (otras frases, más ruido, otras probabilidades)

Las proporciones globales se calibran con los conteos agregados del piloto de la
UPC (31 días, 120 conversaciones): 36 % terminó con asesor y 23 % de los mensajes
no tuvo atajo. No se usa ningún texto ni etiqueta del piloto.

Uso:  python -m src.generar_datos --seed 42
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .observaciones import observar, ruta_atajo
from .texto import ensuciar

DATA = Path(__file__).resolve().parent.parent / "data"

INTENCIONES = ["estado_estudio", "horarios", "sedes", "pagos", "preparacion", "resultados_envio", "otro"]
P_INTENCION = [0.35, 0.15, 0.12, 0.13, 0.10, 0.10, 0.05]
# Probabilidad de que el caso sea complejo (necesita una persona) por intención.
P_COMPLEJO = {"estado_estudio": 0.42, "horarios": 0.05, "sedes": 0.05, "pagos": 0.52,
              "preparacion": 0.12, "resultados_envio": 0.40, "otro": 0.60}

# ---------------------------------------------------------------- plantillas
# Cada variante tiene sus propias frases. {quien} = "mi" / "el de mi mamá"...
PLANTILLAS = {
    "G1": {
        "apertura": {
            "estado_estudio": ["Buenos días, quería saber si ya está listo {quien} resultado", "Hola, ¿mi examen ya salió?",
                               "Buenas tardes, consulto por {quien} estudio", "Hola, ¿ya tienen {quien} resultado de la muestra?"],
            "horarios": ["Hola, ¿cuál es el horario de atención?", "¿A qué hora abren el sábado?", "Buenas, ¿atienden los domingos?"],
            "sedes": ["¿Dónde queda la sede del centro?", "Hola, ¿qué sedes tienen?", "Buenas, necesito la dirección de la sede"],
            "pagos": ["¿Cuánto cuesta el examen de sangre?", "Hola, ¿puedo pagar por transferencia?", "Buenas, ¿qué valor tiene el perfil lipídico?"],
            "preparacion": ["¿El examen es en ayunas?", "Hola, ¿qué preparación necesito para el examen de orina?", "¿Cuántas horas de ayuno debo hacer?"],
            "resultados_envio": ["Hola, no me ha llegado {quien} resultado al correo", "Buenas, ¿me pueden enviar {quien} resultado?",
                                 "Quisiera que me reenviaran los resultados"],
            "otro": ["Hola, necesito una certificación para mi empresa", "Buenas, tengo una duda con una factura",
                     "Hola, quiero cambiar el correo registrado"],
        },
        "detalle_complejo": {
            "estado_estudio": ["Me dijeron que la muestra fue rechazada y no sé qué hacer", "Ya pasaron más de 8 días y nada",
                               "En el sistema aparece pendiente desde la semana pasada", "Necesito saber por qué se demora tanto"],
            "pagos": ["Ya hice la transferencia y no aparece el pago", "Me cobraron dos veces", "El valor que me dijeron no coincide con la factura"],
            "resultados_envio": ["El correo que tienen está mal escrito", "Me llegó el resultado de otra persona",
                                 "Lo necesito para una cita médica mañana"],
            "otro": ["Es para un trámite de la empresa y me lo exigen hoy", "La factura tiene los datos equivocados",
                     "Necesito corregir mis datos en el sistema"],
            "_default": ["No, eso no responde mi pregunta", "Mi caso es diferente", "Tengo un problema con eso"],
        },
        "seguimiento_simple": ["¿Y también el sábado?", "Ok, ¿y en la otra sede?", "¿Necesito llevar la orden?", "¿Y el precio incluye el examen de orina?",
                               "¿Puede reclamarlo otra persona por mí?", "Es urgente, ¿hoy alcanzo a llegar?", "Otra vez gracias, ¿y el domingo?"],
        "pide_asesor": ["Quiero hablar con un asesor", "¿Me puede atender una persona?", "Por favor comuníquenme con alguien real",
                        "Necesito un asesor ya"],
        "negativo": ["Esto no sirve, llevo días esperando", "Qué falta de seriedad", "Ya les dije lo mismo, no entiende",
                     "Es urgente, estoy molesto", "Voy a poner una queja"],
        "cierre": ["Gracias", "Muchas gracias", "Listo, gracias", "Perfecto, muy amable"],
        "otro_texto": ["Ok", "Mmm", "No entiendo", "¿Cómo así?", "Sí"],
    },
    "G2": {
        "apertura": {
            "estado_estudio": ["Buen día, ¿me confirman si ya procesaron {quien} examen?", "Necesito saber cómo va {quien} estudio",
                               "¿Ya publicaron {quien} resultado?", "Hola, consulto por {quien} examen de ayer"],
            "horarios": ["¿Hasta qué hora atienden hoy?", "¿Abren en festivo?", "¿El horario del sábado cuál es?"],
            "sedes": ["¿Tienen sucursal en el norte?", "¿Cuál es la ubicación del laboratorio?", "¿Dónde queda la sede más cercana?"],
            "pagos": ["¿Aceptan tarjeta?", "¿Qué precio tiene la prueba de embarazo?", "¿Se puede pagar por Nequi?"],
            "preparacion": ["¿Hay que ir en ayunas para la glucosa?", "¿Cómo debo prepararme para el examen?", "¿Puedo tomar agua antes?"],
            "resultados_envio": ["{quien} resultado nunca llegó", "¿Por qué medio entregan los resultados?", "Requiero copia {quien} resultado"],
            "otro": ["Necesito una constancia de asistencia", "Tengo un inconveniente con mi factura electrónica",
                     "Quiero actualizar mis datos"],
        },
        "detalle_complejo": {
            "estado_estudio": ["Dicen que hay que repetir la toma y nadie me explica", "Ya va casi una semana sin respuesta",
                               "La página dice en proceso hace días", "Me toca viajar y lo necesito antes"],
            "pagos": ["Pagué en línea y figura sin pagar", "Me descontaron doble", "La cifra cobrada es distinta a la cotizada"],
            "resultados_envio": ["El email registrado no es el mío", "El documento que llegó no es mío", "Mi médico lo pide para mañana temprano"],
            "otro": ["La constancia la piden hoy mismo", "El NIT en la factura está errado", "Mis datos quedaron mal registrados"],
            "_default": ["Eso no era lo que pregunté", "No es mi caso", "Sigo con el mismo inconveniente"],
        },
        "seguimiento_simple": ["¿Y los domingos?", "¿Hay parqueadero?", "¿Debo llevar documento?", "¿Cuánto se demora?",
                               "¿Tengo que ir en persona?", "Mi mamá está molesta porque no sabía del ayuno, ¿cuántas horas son?",
                               "Disculpe que pregunte de nuevo lo mismo, ¿abren festivos?"],
        "pide_asesor": ["Páseme con un funcionario", "¿Hay algún humano que me ayude?", "Necesito un agente por favor", "Asesora por favor"],
        "negativo": ["Pésimo servicio", "Esto es inaceptable", "Nadie responde nada", "Otra vez lo mismo, no me ayuda", "Estoy cansada de escribir"],
        "cierre": ["Ok gracias", "Vale gracias", "Gracias", "Muy amable"],
        "otro_texto": ["Ajá", "¿Qué?", "No me queda claro", "Bueno", "Hmm"],
    },
}

PARAMS = {
    #           mala escr.  tasa ruido  nota voz  p pide  p neg  p repite  umbral frustr.
    "G1": dict(p_mala=0.25, ruido=0.35, p_voz=0.06, p_pide=0.30, p_neg=0.30, p_rep=0.25, umbral=3.0),
    "G2": dict(p_mala=0.30, ruido=0.45, p_voz=0.09, p_pide=0.24, p_neg=0.32, p_rep=0.28, umbral=3.0),
}


def _elige(rng, lista):
    return lista[int(rng.integers(0, len(lista)))]


def generar_conversacion(rng, cid: str, variante: str) -> tuple[dict, list[dict]]:
    T, P = PLANTILLAS[variante], PARAMS[variante]
    intencion = INTENCIONES[int(rng.choice(len(INTENCIONES), p=P_INTENCION))]
    tercero = rng.random() < 0.40
    mala = rng.random() < P["p_mala"]
    # Un tercero (familiar) que consulta estudios suele necesitar autorización.
    p_cx = P_COMPLEJO[intencion] + (0.10 if tercero and intencion in ("estado_estudio", "resultados_envio") else 0.0)
    complejo = rng.random() < p_cx
    turno_revela = int(rng.integers(2, 5)) if complejo else None
    familiar = _elige(rng, ["mi mamá", "mi hijo", "mi esposa", "mi papá"]) if tercero else None

    def redactar(txt):
        if familiar:
            txt = re.sub(r"\{quien\} (\w+)", lambda m: f"el {m.group(1)} de {familiar}", txt)
            txt = txt.replace(" de el ", " del ").replace("{quien}", "")
        txt = txt.replace("{quien}", "mi")
        txt = txt[0].upper() + txt[1:] if txt else txt
        return ensuciar(txt, rng, P["ruido"]) if mala else txt

    frustracion, oraculo, necesita = 0.0, None, False
    turnos, previo, n_max = [], None, int(rng.integers(5, 11))
    for t in range(1, n_max + 1):
        es_voz, tipo = False, "benigno"
        if t == 1:
            texto = redactar(_elige(rng, T["apertura"][intencion]))
        elif necesita:
            r = rng.random()
            tipo = "fallo"
            if r < P["p_pide"] * (0.7 if tercero else 1.0):
                texto = redactar(_elige(rng, T["pide_asesor"]))
            elif r < P["p_pide"] + P["p_neg"]:
                texto = redactar(_elige(rng, T["negativo"]))
            elif r < P["p_pide"] + P["p_neg"] + P["p_rep"] and previo:
                texto = previo                                    # repite el pedido
            else:
                texto = redactar(_elige(rng, T["detalle_complejo"].get(intencion, T["detalle_complejo"]["_default"])))
        elif complejo and t >= turno_revela:
            tipo = "fallo"
            texto = redactar(_elige(rng, T["detalle_complejo"].get(intencion, T["detalle_complejo"]["_default"])))
        else:
            r = rng.random()
            if not complejo and (r < 0.45 or (t >= 3 and frustracion < 1)):
                texto = redactar(_elige(rng, T["cierre"]))
            elif r < 0.80:
                texto = redactar(_elige(rng, T["seguimiento_simple"]))
            else:
                texto = redactar(_elige(rng, T["otro_texto"]))
        if rng.random() < P["p_voz"] * (1.6 if mala else 1.0):
            es_voz, texto = True, "[NOTA_DE_VOZ]"

        sin_atajo = es_voz or ruta_atajo(texto) is None
        o = observar(texto, es_voz, previo)
        # Dinámica latente de frustración (no observable para el modelo).
        if o["obs"] == "CIERRE":
            frustracion = max(0.0, frustracion - 1.0)
        else:
            peso = 1.0 if tipo == "fallo" else (0.5 if es_voz else 0.35)   # "Ok", "Sí" o una nota de voz no son un fallo real
            frustracion += peso * sin_atajo + 0.8 * o["s_repite"] + (0.5 if complejo and t >= (turno_revela or 99) else 0.0)
            if o["obs"] == "ATAJO":
                frustracion = max(0.0, frustracion - 0.4)
        if not necesita and ((complejo and t >= turno_revela) or frustracion >= P["umbral"]):
            necesita, oraculo = True, t
        estado = 3 if necesita else (2 if frustracion >= 1.0 else 1)
        turnos.append({"conversacion_id": cid, "turno": t, "texto": texto, "es_voz": int(es_voz),
                       "estado_latente": estado, "frustracion": round(frustracion, 2), **o})
        previo = None if es_voz else texto
        if o["obs"] == "CIERRE" and not necesita:
            break
        if necesita and t - oraculo >= int(rng.integers(1, 4)):
            break   # en la realidad el caso termina con un asesor o el usuario abandona
    conv = {"conversacion_id": cid, "variante": variante, "intencion": intencion,
            "tipo_usuario": "tercero" if tercero else "titular", "escritura": "mala" if mala else "buena",
            "usa_voz": int(any(x["es_voz"] for x in turnos)), "caso_complejo": int(complejo),
            "requiere_asesor": int(necesita), "turno_oraculo": oraculo, "n_turnos": len(turnos)}
    return conv, turnos


def generar(n: int, seed: int, variante: str):
    rng = np.random.default_rng(seed)
    convs, turnos = [], []
    for i in range(n):
        c, ts = generar_conversacion(rng, f"{variante}-{seed}-{i:05d}", variante)
        convs.append(c)
        turnos.extend(ts)
    return pd.DataFrame(convs), pd.DataFrame(turnos)


def particionar(convs: pd.DataFrame, seed: int, frac=(0.70, 0.15, 0.15)):
    """Partición por conversación (nunca por mensaje) para evitar fuga."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(convs))
    n_tr, n_va = int(frac[0] * len(convs)), int(frac[1] * len(convs))
    split = np.empty(len(convs), dtype=object)
    split[idx[:n_tr]], split[idx[n_tr:n_tr + n_va]], split[idx[n_tr + n_va:]] = "train", "val", "test_g1"
    return split


def construir(seed: int, n_g1: int = 2000, n_g2: int = 300, out: Path = DATA, guardar: bool = True):
    c1, t1 = generar(n_g1, seed, "G1")
    c1["particion"] = particionar(c1, seed)
    c2, t2 = generar(n_g2, seed + 1000, "G2")
    c2["particion"] = "test_g2"
    convs, turnos = pd.concat([c1, c2], ignore_index=True), pd.concat([t1, t2], ignore_index=True)
    if guardar:
        d = out / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        convs.to_csv(d / "conversaciones.csv", index=False)
        turnos.to_csv(d / "turnos.csv", index=False)
    return convs, turnos


def resumen(convs, turnos) -> dict:
    g1 = convs[convs.variante == "G1"]
    tg1 = turnos[turnos.conversacion_id.isin(g1.conversacion_id)]
    return {"conversaciones": int(len(convs)), "turnos": int(len(turnos)),
            "pct_requiere_asesor_G1": round(100 * g1.requiere_asesor.mean(), 1),
            # "Cayó al LLM": sin atajo y sin regla de escalamiento que lo atrape
            # (las solicitudes de asesor y el léxico negativo los maneja una regla).
            "pct_mensajes_al_llm_G1": round(100 * (tg1.s_sin_atajo.astype(bool)
                                                   & ~tg1.obs.isin(["PIDE_ASESOR", "NEGATIVO"])).mean(), 1),
            "objetivo_piloto": {"pct_con_asesor": 36, "pct_mensajes_sin_atajo": 23}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=2000, help="conversaciones G1 (train/val/test)")
    ap.add_argument("--n-g2", type=int, default=300, help="conversaciones de prueba G2")
    a = ap.parse_args()
    convs, turnos = construir(a.seed, a.n, a.n_g2)
    print(json.dumps(resumen(convs, turnos), ensure_ascii=False, indent=2))
