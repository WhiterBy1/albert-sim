"""Decisores de escalamiento: líneas base y HMM con filtrado hacia adelante.

Todos reciben las observaciones de una conversación turno a turno y devuelven,
para cada turno, si escalan y (cuando aplica) la probabilidad de "requiere asesor".
Ninguno ve el futuro de la conversación: deciden en línea, como lo haría Albert.
"""
import numpy as np
import pandas as pd

from .observaciones import IDX, SIMBOLOS

N_EST, N_OBS = 3, len(SIMBOLOS)
FALLOS = {"SIN_ATAJO", "VOZ", "REPITE"}


class Mayoritaria:
    """Línea base 2: predice siempre la clase más frecuente del entrenamiento."""
    nombre = "Línea base 2 · Clase mayoritaria"

    def ajustar(self, convs, turnos):
        self.prior = float(convs.requiere_asesor.mean())
        self.clase = int(self.prior >= 0.5)
        return self

    def decidir(self, obs: list[str], **_):
        return [self.clase] * len(obs), [self.prior] * len(obs)


class ReglasFijas:
    """Línea base 1: lógica actual de Albert.
    Escala si el usuario pide un asesor, usa léxico negativo o acumula 3 fallos seguidos."""
    nombre = "Línea base 1 · Reglas fijas (Albert)"

    def __init__(self, fallos_seguidos: int = 3):
        self.k = fallos_seguidos

    def ajustar(self, convs, turnos):
        return self

    def decidir(self, obs, **_):
        racha, dec = 0, []
        for o in obs:
            racha = racha + 1 if o in FALLOS else 0
            dec.append(int(o in ("PIDE_ASESOR", "NEGATIVO") or racha >= self.k))
        return dec, [np.nan] * len(obs)


class HMMEscalamiento:
    """(A) HMM de 3 estados ocultos (encaminado, atascado, requiere asesor).

    Parámetros por máxima verosimilitud con suavizado de Laplace a partir de las
    secuencias de estado del entrenamiento; decisión por filtrado hacia adelante:
        f_{1:t+1} = alfa * O_{t+1} A^T f_{1:t},   escalar si P(S3 | o_{1:t}) >= tau
    """
    nombre = "(A) HMM · filtrado hacia adelante"

    def __init__(self, tau: float = 0.20, alfa_laplace: float = 1.0):
        self.tau, self.a = tau, alfa_laplace

    def ajustar(self, convs, turnos):
        pi = np.full(N_EST, self.a)
        A = np.full((N_EST, N_EST), self.a)
        B = np.full((N_EST, N_OBS), self.a)
        for _, g in turnos.groupby("conversacion_id", sort=False):
            s = g.estado_latente.values - 1
            o = g.obs.map(IDX).values
            pi[s[0]] += 1
            np.add.at(A, (s[:-1], s[1:]), 1)
            np.add.at(B, (s, o), 1)
        self.pi = pi / pi.sum()
        self.A = A / A.sum(1, keepdims=True)
        self.B = B / B.sum(1, keepdims=True)
        return self

    def filtrar(self, obs):
        f, probs = None, []
        for o in obs:
            e = self.B[:, IDX[o]]
            f = e * (self.pi if f is None else self.A.T @ f)
            f = f / f.sum()
            probs.append(float(f[2]))
        return probs

    def decidir(self, obs, **_):
        p = self.filtrar(obs)
        return [int(x >= self.tau) for x in p], p


def aplicar(modelo, convs: pd.DataFrame, turnos: pd.DataFrame) -> pd.DataFrame:
    """Corre el decisor en línea y resume por conversación: escala, turno y score.
    El score es la probabilidad máxima alcanzada hasta el turno de escalar
    (o en toda la conversación si nunca escala)."""
    filas = []
    por_conv = dict(tuple(turnos.groupby("conversacion_id", sort=False)))
    for _, c in convs.iterrows():
        g = por_conv[c.conversacion_id]
        dec, prob = modelo.decidir(g.obs.tolist(), tercero=int(c.get("tipo_usuario") == "tercero"))
        t_esc = next((i + 1 for i, d in enumerate(dec) if d), None)
        hasta = t_esc or len(dec)
        score = float(np.nanmax(prob[:hasta])) if not np.all(np.isnan(prob)) else np.nan
        filas.append({**c.to_dict(), "escala": int(t_esc is not None), "turno_escala": t_esc, "score": score})
    return pd.DataFrame(filas)


class RegresionLogistica:
    """(B) Regresión logística L2 sin estado oculto: estima P(requiere asesor | rasgos del turno y de la ventana).

    Rasgos en el turno t (solo información hasta t): símbolo actual (one-hot), conteo de cada símbolo
    en los últimos k turnos y acumulado en la conversación, índice del turno, racha de fallos y tipo
    de usuario (tercero). Etiqueta de entrenamiento por turno: el estado latente es «requiere asesor».
    """
    nombre = "(B) Regresión logística · ventana de turnos"

    def __init__(self, c: float = 1.0, k: int = 3, tau: float = 0.20):
        self.c, self.k, self.tau = c, k, tau

    def rasgos(self, obs, tercero=0):
        t = len(obs)
        oh = np.zeros((t, N_OBS))
        oh[np.arange(t), [IDX[o] for o in obs]] = 1.0
        acum = np.cumsum(oh, axis=0)
        ventana = acum - np.vstack([np.zeros((self.k, N_OBS)), acum[:-self.k]])[:t] if t > self.k else acum.copy()
        racha, r = np.zeros(t), 0
        for i, o in enumerate(obs):
            r = r + 1 if o in FALLOS else 0
            racha[i] = r
        return np.column_stack([oh, ventana, acum, np.arange(1, t + 1), racha, np.full(t, float(tercero))])

    def ajustar(self, convs, turnos):
        from sklearn.linear_model import LogisticRegression
        tipo = dict(zip(convs.conversacion_id, (convs.tipo_usuario == "tercero").astype(int)))
        X, y = [], []
        for cid, g in turnos.groupby("conversacion_id", sort=False):
            X.append(self.rasgos(g.obs.tolist(), tipo.get(cid, 0)))
            y.append((g.estado_latente.values == 3).astype(int))
        self.mu = np.vstack(X).mean(0)
        self.sd = np.vstack(X).std(0) + 1e-9
        self.m = LogisticRegression(C=self.c, max_iter=2000).fit((np.vstack(X) - self.mu) / self.sd, np.concatenate(y))
        return self

    def decidir(self, obs, tercero=0, **_):
        p = self.m.predict_proba((self.rasgos(obs, tercero) - self.mu) / self.sd)[:, 1]
        return [int(x >= self.tau) for x in p], [float(x) for x in p]


TAUS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]


def tau_en_validacion(modelo, convs_val, turnos_val, recall_min=0.90):
    """Umbral que maximiza F1 en validación con recall >= recall_min (si ninguno lo cumple, el de mayor recall).
    Desvío del Hito 1 (que solo permitía bajar τ): se informa aparte como variante «τ en validación»."""
    from .metricas import evaluar
    mejor, tau0 = None, modelo.tau
    for tau in TAUS:
        modelo.tau = tau
        r = evaluar(aplicar(modelo, convs_val, turnos_val))
        clave = (r["recall"] >= recall_min, r["f1"] if r["recall"] >= recall_min else r["recall"])
        if mejor is None or clave > mejor[0]:
            mejor = (clave, tau)
    modelo.tau = tau0
    return mejor[1]


def ajustar_tau(modelo: HMMEscalamiento, convs_val, turnos_val, recall_min=0.90, paso=0.01):
    """Regla predeclarada: se parte de tau* = C_FP/(C_FP+C_FN) = 0.20 y solo se baja
    si en validación no se alcanza Recall >= 0.90."""
    from .metricas import evaluar
    tau = modelo.tau
    while tau > paso:
        modelo.tau = tau
        if evaluar(aplicar(modelo, convs_val, turnos_val))["recall"] >= recall_min:
            break
        tau = round(tau - paso, 4)
    modelo.tau = tau
    return tau
