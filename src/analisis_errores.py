"""Análisis de errores del modelo propuesto (HMM con τ en validación) en la prueba G2 de la semilla 42.

Clasifica cada falso negativo (FN) y falso positivo (FP) por tipo, con una hipótesis de causa, y guarda:
  experiments/errores_por_tipo.csv       conteo por tipo (todos los errores)
  experiments/errores_muestra.csv        todos los errores del HMM (A') y 4 FP de la logística, con trayectoria de símbolos y mensajes
Las hipótesis son eso, hipótesis: se validan leyendo las conversaciones de la muestra.

Uso:  python -m src.analisis_errores
"""
import pandas as pd

from .modelos import aplicar
from .run_experimentos import EXP, N_HMM_V, N_LR_V, cargar, entrenar_modelos, sub

SEED = 42
HIPOTESIS = {
    "FN_sin_senal_explicita": "El usuario necesitaba asesor pero nunca lo pidió, ni usó léxico negativo ni repitió: solo mensajes «sin atajo». "
                              "El HMM necesita señales fuertes para subir P(S3); con SIN_ATAJO aislados la creencia queda bajo τ.",
    "FN_conversacion_corta": "La conversación terminó pocos turnos después de que la necesidad se hizo evidente: no hubo tiempo de acumular evidencia.",
    "FN_voz": "Las notas de voz llegan como «[NOTA_DE_VOZ]» sin texto: el sensor no distingue si el contenido era un fallo o una queja.",
    "FN_otro": "Evidencia débil y repartida; la probabilidad filtrada se queda cerca de τ sin cruzarlo (decisión de frontera).",
    "FP_pide_asesor_sin_necesidad": "El usuario pidió un asesor sin que el caso lo requiriera según la etiqueta (curiosidad o preferencia por hablar con una persona); "
                                    "la solicitud explícita eleva mucho P(S3). Posible error de etiqueta o política que conviene decidir con la UPC.",
    "FP_lexico_negativo": "Léxico de enojo o urgencia en un caso resoluble («urgente», «otra vez»): el sensor lo trata como señal de frustración.",
    "FP_repeticion": "Mensajes repetidos (Jaccard ≥ 0,5) sin que el usuario estuviera frustrado: reformular el mismo dato se confunde con insistir.",
    "FP_voz": "Nota de voz seguida de otros turnos sin atajo: la voz cuenta como fallo parcial y suma evidencia.",
    "FP_sin_atajo_acumulado": "Varios mensajes «sin atajo» seguidos (frases cortas como «ok» o «sí» que no activan una ruta) acumulan evidencia sin haber frustración.",
}


def clasificar(fila, simbolos: list[str]):
    s = set(simbolos)
    if fila.requiere_asesor == 1:      # falso negativo
        if fila.usa_voz == 1:
            return "FN_voz"
        if not s & {"PIDE_ASESOR", "NEGATIVO", "REPITE"}:
            return "FN_sin_senal_explicita"
        if fila.n_turnos - (fila.turno_oraculo or 0) <= 1:
            return "FN_conversacion_corta"
        return "FN_otro"
    if "PIDE_ASESOR" in s:
        return "FP_pide_asesor_sin_necesidad"
    if "NEGATIVO" in s:
        return "FP_lexico_negativo"
    if "REPITE" in s:
        return "FP_repeticion"
    if "VOZ" in s:
        return "FP_voz"
    return "FP_sin_atajo_acumulado"


def main():
    convs, turnos = cargar(SEED)
    modelos = {m.nombre: m for m in entrenar_modelos(sub(convs, turnos, "train"), sub(convs, turnos, "val"))}
    c, t = sub(convs, turnos, "test_g2")
    por_conv = dict(tuple(t.groupby("conversacion_id", sort=False)))
    todos = []
    for nombre in (N_HMM_V, N_LR_V):
        res = aplicar(modelos[nombre], c, t)
        err = res[res.requiere_asesor != res.escala]
        for r in err.itertuples():
            g = por_conv[r.conversacion_id]
            simbolos = g.obs.tolist()
            tipo = clasificar(r, simbolos)
            todos.append({"modelo": nombre, "conversacion_id": r.conversacion_id, "error": "FN" if r.requiere_asesor else "FP",
                          "tipo_error": tipo, "intencion": r.intencion, "tipo_usuario": r.tipo_usuario, "escritura": r.escritura,
                          "usa_voz": r.usa_voz, "n_turnos": r.n_turnos, "turno_oraculo": r.turno_oraculo,
                          "turno_escala": r.turno_escala, "p_max": round(r.score, 3),
                          "trayectoria": " > ".join(simbolos), "mensajes": " | ".join(g.texto.tolist()),
                          "hipotesis_por_validar": HIPOTESIS[tipo]})
    df = pd.DataFrame(todos)
    (df.groupby(["modelo", "error", "tipo_error"]).size().rename("casos").reset_index()
       .to_csv(EXP / "errores_por_tipo.csv", index=False))
    hmm = df[df.modelo == N_HMM_V]
    lr = df[df.modelo == N_LR_V]
    muestra = pd.concat([hmm, lr[lr.error == "FP"].head(4)])      # todos los errores del HMM + 4 FP de la logística
    muestra.to_csv(EXP / "errores_muestra.csv", index=False)
    print(df.groupby(["modelo", "error", "tipo_error"]).size().to_string())
    print(f"\nMuestra para validar leyendo: {len(muestra)} casos -> experiments/errores_muestra.csv")


if __name__ == "__main__":
    main()
