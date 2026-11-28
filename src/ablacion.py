"""Sensibilidad (ablación) en la prueba G2, 5 semillas:
  * reglas de Albert con k = 2, 3, 4 fallos seguidos;
  * regresión logística con ventana k = 1, 3, 5 (C = 0,1; τ elegido en validación);
  * HMM con τ fijo en 0,20 / 0,50 / 0,75 (solo para mostrar el compromiso; τ no se elige con la prueba).
Uso:  python -m src.ablacion     ->  experiments/ablacion_g2.csv
"""
import pandas as pd

from .metricas import evaluar
from .modelos import HMMEscalamiento, RegresionLogistica, ReglasFijas, aplicar, tau_en_validacion
from .run_experimentos import EXP, cargar, sub

filas = []
for seed in [42, 43, 44, 45, 46]:
    c, t = cargar(seed)
    tr, va, te = sub(c, t, "train"), sub(c, t, "val"), sub(c, t, "test_g2")
    variantes = [(f"Reglas · k = {k} fallos", ReglasFijas(fallos_seguidos=k)) for k in (2, 3, 4)]
    for k in (1, 3, 5):
        m = RegresionLogistica(c=0.1, k=k).ajustar(*tr)
        m.tau = tau_en_validacion(m, *va)
        variantes.append((f"Logística · ventana k = {k}", m))
    h = HMMEscalamiento(alfa_laplace=0.1).ajustar(*tr)
    for tau in (0.20, 0.50, 0.75):
        hh = HMMEscalamiento(alfa_laplace=0.1, tau=tau); hh.pi, hh.A, hh.B = h.pi, h.A, h.B
        variantes.append((f"HMM · τ = {tau:.2f}", hh))
    for nombre, m in variantes:
        r = evaluar(aplicar(m, *te))
        filas.append({"seed": seed, "variante": nombre, **{k: r[k] for k in ("recall", "f1", "pct_innecesarios", "retraso_mediana")}})
df = pd.DataFrame(filas)
EXP.mkdir(exist_ok=True)
agg = df.groupby("variante", sort=False)[["recall", "f1", "pct_innecesarios", "retraso_mediana"]].agg(["mean", "std"]).round(3)
df.to_csv(EXP / "ablacion_g2_por_semilla.csv", index=False)
agg.to_csv(EXP / "ablacion_g2.csv")
print(agg.to_string())
