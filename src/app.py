"""Prototipo funcional de Albert-sim (Streamlit).

Pestañas: (1) chat en vivo con la decisión de las reglas, del HMM y de la regresión logística turno a turno;
(2) conversaciones de la base PostgreSQL con las curvas de probabilidad; (3) resultados de los experimentos.
Todo corre localmente: no se envía ningún dato a una API externa. Datos 100 % sintéticos.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.cargar_db import URL
from src.metricas import UMBRALES
from src.modelos import ReglasFijas, aplicar
from src.observaciones import observar, ruta_atajo
from src.run_experimentos import EXP, N_HMM_V, N_LR_V, cargar, entrenar_modelos, sub

SEED = 42
CACHE_MODELOS = Path(__file__).resolve().parent.parent / ".cache" / "modelos_seed42.pkl"
st.set_page_config(page_title="Albert-sim", page_icon="🩺", layout="wide")


@st.cache_data(show_spinner="Leyendo conversaciones...")
def leer_datos():
    """Lee de PostgreSQL; si no está disponible (ejecución sin Docker) usa los CSV generados."""
    try:
        eng = create_engine(URL)
        convs = pd.read_sql(f"SELECT * FROM conversacion WHERE seed = {SEED}", eng)
        turnos = pd.read_sql("SELECT m.* FROM mensaje m JOIN conversacion c USING (conversacion_id) "
                             f"WHERE c.seed = {SEED} ORDER BY m.conversacion_id, m.turno", eng)
        if len(convs):
            return convs, turnos, "PostgreSQL"
    except Exception:
        pass
    convs, turnos = cargar(SEED)
    return convs, turnos, "CSV local"


@st.cache_resource(show_spinner="Entrenando HMM y regresión logística (solo con entrenamiento/validación)...")
def modelos():
    if CACHE_MODELOS.exists():
        try:
            with CACHE_MODELOS.open("rb") as archivo:
                return pickle.load(archivo)
        except Exception:
            CACHE_MODELOS.unlink(missing_ok=True)

    convs, turnos, _ = leer_datos()
    ms = {m.nombre: m for m in entrenar_modelos(sub(convs, turnos, "train"), sub(convs, turnos, "val"))}
    CACHE_MODELOS.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_MODELOS.open("wb") as archivo:
        pickle.dump(ms, archivo)
    return ms


def veredicto(obs, tercero):
    ms = modelos()
    reglas, _ = ReglasFijas().decidir(obs)
    hmm, p_h = ms[N_HMM_V].decidir(obs, tercero=tercero)
    lr, p_l = ms[N_LR_V].decidir(obs, tercero=tercero)
    return reglas, hmm, p_h, lr, p_l


convs, turnos, fuente = leer_datos()
st.title("Albert-sim · ¿cuándo pasar la conversación a un asesor?")
st.caption(f"Proyecto Final de Inteligencia Artificial (UTB) · Ruta B: razonamiento con incertidumbre · datos sintéticos desde {fuente} "
           f"({len(convs)} conversaciones, semilla {SEED}) · sin llamadas a APIs externas")
ms = modelos()
tau_h, tau_l = ms[N_HMM_V].tau, ms[N_LR_V].tau
t1, t2, t3 = st.tabs(["💬 Chat en vivo", "🗂️ Conversaciones de la base", "📊 Resultados"])

# ---------------------------------------------------------------- 1. chat en vivo
with t1:
    st.write("Escribe como lo haría un paciente. El sensor convierte cada mensaje en una observación y los tres decisores "
             "actualizan su estimación turno a turno. **El texto no sale de este equipo.**")
    if "msgs" not in st.session_state:
        st.session_state.msgs = []          # lista de (texto, es_voz)
    c1, c2, c3 = st.columns([4, 1, 1])
    tercero = c3.checkbox("Escribe un familiar (tercero)")
    voz = c2.checkbox("Nota de voz")
    texto = c1.text_input("Mensaje del usuario", key="entrada", placeholder="Ej.: ¿Ya está listo el resultado de mi mamá?")
    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("Enviar") and (texto.strip() or voz):
        st.session_state.msgs.append(("[NOTA_DE_VOZ]" if voz else texto.strip(), voz))
    if b2.button("Reiniciar"):
        st.session_state.msgs = []
    if st.session_state.msgs:
        obs, previo, filas = [], None, []
        for txt, es_voz in st.session_state.msgs:
            o = observar(txt, es_voz, previo)
            obs.append(o["obs"])
            previo = None if es_voz else txt
            filas.append({"turno": len(obs), "mensaje": txt, "ruta": ruta_atajo(txt) or "(sin atajo → LLM)", "observación": o["obs"]})
        reglas, hmm, p_h, lr, p_l = veredicto(obs, int(tercero))
        for i, f in enumerate(filas):
            f.update({"reglas actuales": "ESCALA" if reglas[i] else "—", f"HMM P(frustración)": round(p_h[i], 3),
                      "HMM": "ESCALA" if hmm[i] else "—", "Logística P": round(p_l[i], 3), "Logística": "ESCALA" if lr[i] else "—"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
        if "PIDE_ASESOR" in obs:
            st.warning("Regla de seguridad (Hito 1, riesgo R2): una solicitud explícita de asesor siempre se escala, aunque el modelo aún no lo haga.")
        a, b, c = st.columns(3)
        a.metric("Reglas actuales (Albert)", "Escalar" if any(reglas) else "Seguir")
        b.metric(f"HMM (τ = {tau_h:.2f})", "Escalar" if any(hmm) else "Seguir", f"P = {p_h[-1]:.2f}")
        c.metric(f"Regresión logística (τ = {tau_l:.2f})", "Escalar" if any(lr) else "Seguir", f"P = {p_l[-1]:.2f}")
        st.line_chart(pd.DataFrame({"HMM": p_h, "Logística": p_l}, index=range(1, len(obs) + 1)))
        st.caption("Prueba con: «¿Ya está listo el resultado de mi mamá?» → «Ya pasaron 8 días y nada» → «Ya les dije lo mismo» → «Quiero hablar con un asesor».")
    else:
        st.info("Aún no hay mensajes.")

# ---------------------------------------------------------------- 2. conversaciones
with t2:
    particion = st.selectbox("Partición", ["test_g2", "test_g1", "val", "train"], index=0)
    c_sel = convs[convs.particion == particion]
    solo = st.radio("Mostrar", ["Todas", "Requieren asesor", "No requieren"], horizontal=True)
    if solo != "Todas":
        c_sel = c_sel[c_sel.requiere_asesor == int(solo == "Requieren asesor")]
    cid = st.selectbox("Conversación", c_sel.conversacion_id.tolist())
    fila = convs[convs.conversacion_id == cid].iloc[0]
    g = turnos[turnos.conversacion_id == cid].sort_values("turno")
    obs = g.obs.tolist()
    reglas, hmm, p_h, lr, p_l = veredicto(obs, int(fila.tipo_usuario == "tercero"))
    st.write(f"**Etiqueta real:** {'requiere asesor' if fila.requiere_asesor else 'no requiere asesor'}"
             f"{'' if pd.isna(fila.turno_oraculo) else f' (evidente desde el turno {int(fila.turno_oraculo)})'} · "
             f"{fila.intencion} · {fila.tipo_usuario} · escritura {fila.escritura}")
    det = g[["turno", "texto", "obs"]].assign(reglas=["ESCALA" if x else "—" for x in reglas], HMM=[round(x, 3) for x in p_h],
                                               logistica=[round(x, 3) for x in p_l])
    st.dataframe(det, hide_index=True, width="stretch")
    curva = pd.DataFrame({"HMM": p_h, "Logística": p_l, f"τ HMM = {tau_h:.2f}": tau_h}, index=g.turno.values)
    st.line_chart(curva)
    st.caption("Por turno, la barra de decisión es la probabilidad filtrada de «requiere asesor»; se escala al cruzar τ.")

# ---------------------------------------------------------------- 3. resultados
with t3:
    st.write("Cifras reproducibles con `docker compose run --rm app python -m src.run_experimentos` (5 semillas + validación cruzada).")
    st.json({"umbrales declarados en el Hito 1": UMBRALES}, expanded=False)
    csv = EXP / "resumen.csv"
    if csv.exists():
        r = pd.read_csv(csv, header=[0, 1], index_col=[0, 1])
        prueba = st.selectbox("Conjunto de prueba", ["test_g2", "test_g1", "val"])
        r = r.loc[prueba]
        plano = pd.DataFrame({m: {k: f"{r.loc[m, (k, 'mean')]:.3f} ± {np.nan_to_num(r.loc[m, (k, 'std')]):.3f}"
                                  for k in ["recall", "f1", "precision", "pct_innecesarios", "retraso_mediana", "ece", "brecha_f1_max"]}
                              for m in r.index}).T
        st.dataframe(plano, width="stretch")
    fig = EXP / "fig_comparacion_test_g2.png"
    if fig.exists():
        st.image(str(fig))
    for nombre, titulo in [("resumen_cv5.csv", "Validación cruzada de 5 particiones"), ("errores_por_tipo.csv", "Errores por tipo (semilla 42, G2)"),
                           ("subgrupos_test_g2.csv", "F1 por subgrupo (G2)")]:
        if (EXP / nombre).exists():
            with st.expander(titulo):
                st.dataframe(pd.read_csv(EXP / nombre, header=[0, 1], index_col=0) if nombre == "resumen_cv5.csv" else pd.read_csv(EXP / nombre),
                             width="stretch")
    if (EXP / "errores_muestra.csv").exists():
        with st.expander("Muestra de errores para validar las hipótesis (16 casos)"):
            st.dataframe(pd.read_csv(EXP / "errores_muestra.csv")[["modelo", "conversacion_id", "tipo_error", "trayectoria", "mensajes",
                                                                    "hipotesis_por_validar"]], width="stretch")
