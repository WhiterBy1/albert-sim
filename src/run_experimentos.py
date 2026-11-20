"""Ejecuta líneas base, HMM y regresión logística con 5 semillas + validación cruzada de 5 particiones.

Uso:  python -m src.run_experimentos               # semillas 42..46 (≈ 1–2 min)
      python -m src.run_experimentos --seeds 42    # una sola semilla (NO reemplaza los resúmenes de 5 semillas)
Salida (carpeta experiments/): seed_<s>.json, resultados_por_semilla.csv, resumen.csv, resumen.md,
       subgrupos_test_g2.csv, cv5_seed42.csv, resumen_cv5.csv, fig_comparacion_test_g2.png

Variantes de umbral (se informan siempre las dos):
  · «τ*»  : regla predeclarada en el Hito 1 (τ* = 0,20; solo se baja si en validación no se llega a recall 0,90).
  · «τ val»: τ elegido en validación (máx. F1 con recall ≥ 0,90). Es un desvío del Hito 1, justificado en el informe:
             con τ* = 0,20 hay demasiadas falsas alarmas (ver experiments/resumen.md).
"""
import argparse
import copy
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .generar_datos import DATA, construir, resumen
from .metricas import SUBGRUPOS, UMBRALES, evaluar
from .modelos import (HMMEscalamiento, Mayoritaria, RegresionLogistica, ReglasFijas, ajustar_tau, aplicar,
                      tau_en_validacion)

EXP = Path(__file__).resolve().parent.parent / "experiments"
CLAVES = ["recall", "f1", "precision", "pct_innecesarios", "retraso_mediana", "ece", "brecha_f1_max"]
GRID_ALFA = [0.1, 0.5, 1.0, 2.0]       # suavizado de Laplace del HMM
GRID_C = [0.01, 0.1, 1.0, 10.0]        # regularización L2 de la logística
N_REGLAS, N_MAYOR = ReglasFijas.nombre, Mayoritaria.nombre
N_HMM, N_HMM_V = "(A) HMM · τ* predeclarado", "(A') HMM · τ en validación"
N_LR, N_LR_V = "(B) Logística · τ* predeclarado", "(B') Logística · τ en validación"
ORDEN = [N_REGLAS, N_MAYOR, N_HMM, N_HMM_V, N_LR, N_LR_V]


def cargar(seed):
    d = DATA / f"seed_{seed}"
    if not (d / "conversaciones.csv").exists():
        construir(seed)
    return pd.read_csv(d / "conversaciones.csv"), pd.read_csv(d / "turnos.csv")


def sub(convs, turnos, particion):
    c = convs[convs.particion == particion] if isinstance(particion, str) else convs[convs.conversacion_id.isin(particion)]
    return c, turnos[turnos.conversacion_id.isin(c.conversacion_id)]


def f1_val(modelo, va):
    return evaluar(aplicar(modelo, *va))["f1"]


def entrenar_modelos(tr, va):
    """Ajusta todo con entrenamiento y elige hiperparámetros y umbrales SOLO con validación."""
    candidatos = [HMMEscalamiento(alfa_laplace=a).ajustar(*tr) for a in GRID_ALFA]
    for m in candidatos:
        ajustar_tau(m, *va)
    hmm = max(candidatos, key=lambda m: f1_val(m, va))
    hmm_v = copy.copy(hmm)
    hmm_v.tau = tau_en_validacion(hmm_v, *va)

    candidatos = [RegresionLogistica(c=c).ajustar(*tr) for c in GRID_C]
    for m in candidatos:
        ajustar_tau(m, *va)
    lr = max(candidatos, key=lambda m: f1_val(m, va))
    lr_v = copy.copy(lr)
    lr_v.tau = tau_en_validacion(lr_v, *va)
    hmm.nombre, hmm_v.nombre, lr.nombre, lr_v.nombre = N_HMM, N_HMM_V, N_LR, N_LR_V
    return [ReglasFijas().ajustar(*tr), Mayoritaria().ajustar(*tr), hmm, hmm_v, lr, lr_v]


def correr_semilla(seed: int) -> dict:
    convs, turnos = cargar(seed)
    tr, va = sub(convs, turnos, "train"), sub(convs, turnos, "val")
    modelos = entrenar_modelos(tr, va)
    hmm, hmm_v, lr, lr_v = modelos[2:]
    out = {"seed": seed, "fecha": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
           "datos": resumen(convs, turnos),
           "hiperparametros": {"hmm_alfa_laplace": hmm.a, "hmm_tau_predeclarado": hmm.tau, "hmm_tau_validacion": hmm_v.tau,
                               "logistica_C": lr.c, "logistica_tau_predeclarado": lr.tau, "logistica_tau_validacion": lr_v.tau},
           "hmm_parametros": {"pi": hmm.pi.round(4).tolist(), "A": hmm.A.round(4).tolist(), "B": hmm.B.round(4).tolist()},
           "resultados": {}}
    for prueba in ("val", "test_g1", "test_g2"):
        c, t = sub(convs, turnos, prueba)
        out["resultados"][prueba] = {m.nombre: evaluar(aplicar(m, c, t)) for m in modelos}
    return out


def validacion_cruzada(seed: int = 42, k: int = 5) -> pd.DataFrame:
    """k particiones por conversación sobre G1. En cada una: se ajusta con 4/5 (80 % entrena, 20 % valida
    hiperparámetros y umbrales) y se evalúa en la partición retenida."""
    convs, turnos = cargar(seed)
    g1 = convs[convs.variante == "G1"].conversacion_id.values
    orden = np.random.default_rng(seed).permutation(g1)
    folds = np.array_split(orden, k)
    filas = []
    for i in range(k):
        test_ids = set(folds[i])
        resto = [c for c in orden if c not in test_ids]
        n_va = int(0.2 * len(resto))
        tr, va, te = sub(convs, turnos, set(resto[n_va:])), sub(convs, turnos, set(resto[:n_va])), sub(convs, turnos, test_ids)
        for m in entrenar_modelos(tr, va):
            r = evaluar(aplicar(m, *te))
            filas.append({"particion": i + 1, "modelo": m.nombre, **{c: r[c] for c in CLAVES}})
    return pd.DataFrame(filas)


def tabla_md(df, titulo):
    agg = df.groupby("modelo", sort=False)[CLAVES].agg(["mean", "std"])
    lineas = [f"## {titulo}", "", "| Modelo | " + " | ".join(CLAVES) + " |", "|---" * (len(CLAVES) + 1) + "|"]
    for modelo in [m for m in ORDEN if m in agg.index]:
        fila = agg.loc[modelo]
        celdas = []
        for k in CLAVES:
            mu, sd = fila[(k, "mean")], fila[(k, "std")]
            celdas.append("—" if np.isnan(mu) else f"{mu:.3f} ± {0 if np.isnan(sd) else sd:.3f}")
        lineas.append(f"| {modelo} | " + " | ".join(celdas) + " |")
    return lineas + [""]


def figura(df_g2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = df_g2.groupby("modelo", sort=False)[["recall", "f1"]].agg(["mean", "std"]).loc[[m for m in ORDEN if m in df_g2.modelo.values]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(agg))
    for j, (k, color, et) in enumerate([("recall", "#86b6ef", "Recall"), ("f1", "#eb6834", "F1")]):
        ax.bar(x + (j - 0.5) * 0.38, agg[(k, "mean")], 0.38, yerr=agg[(k, "std")].fillna(0), label=et, color=color, capsize=2)
    ax.axhline(UMBRALES["recall"], color="#2a5d9f", ls="--", lw=0.8)
    ax.axhline(UMBRALES["f1"], color="#b34a1a", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" · ", "\n", 1) for m in agg.index], fontsize=6)
    ax.set_ylim(0, 1.08)
    ax.set_title("Prueba independiente G2 (5 semillas). Líneas: umbrales recall 0,90 y F1 0,80", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP / "fig_comparacion_test_g2.png", dpi=150)
    plt.close(fig)


def main(seeds):
    EXP.mkdir(exist_ok=True)
    completo = list(seeds) == [42, 43, 44, 45, 46]
    filas, sub_filas = [], []
    for s in seeds:
        r = correr_semilla(s)
        (EXP / f"seed_{s}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        for prueba, por_modelo in r["resultados"].items():
            for modelo, m in por_modelo.items():
                filas.append({"seed": s, "prueba": prueba, "modelo": modelo, **{k: m[k] for k in CLAVES}})
                if prueba == "test_g2":
                    for g in SUBGRUPOS:
                        for nivel, f1 in m[f"f1_por_{g}"].items():
                            sub_filas.append({"seed": s, "modelo": modelo, "subgrupo": g, "nivel": nivel, "f1": f1})
        h = r["hiperparametros"]
        print(f"semilla {s}: alfa={h['hmm_alfa_laplace']} tau_val_hmm={h['hmm_tau_validacion']} C={h['logistica_C']} tau_val_lr={h['logistica_tau_validacion']}")
    df = pd.DataFrame(filas)
    if not completo:
        print("Corrida parcial: no se sobrescriben los resúmenes de 5 semillas.")
        print(df[df.prueba == "test_g2"][["seed", "modelo", "recall", "f1", "precision", "ece"]].round(3).to_string(index=False))
        return
    df.to_csv(EXP / "resultados_por_semilla.csv", index=False)
    df.groupby(["prueba", "modelo"], sort=False)[CLAVES].agg(["mean", "std"]).to_csv(EXP / "resumen.csv")
    (pd.DataFrame(sub_filas).groupby(["modelo", "subgrupo", "nivel"], sort=False).f1.agg(["mean", "std"]).round(3)
     .to_csv(EXP / "subgrupos_test_g2.csv"))
    lineas = ["# Resultados (media ± desv. estándar, 5 semillas)", "",
              f"Semillas: {', '.join(map(str, seeds))}. Umbrales de éxito (Hito 1): {json.dumps(UMBRALES)}", "",
              "τ* = regla predeclarada (0,20; solo baja si en validación recall < 0,90). «τ en validación» = desvío justificado "
              "(máx. F1 con recall ≥ 0,90 en validación). Los hiperparámetros (α del HMM, C de la logística) se eligen solo en validación.", ""]
    titulos = {"test_g2": "Prueba independiente G2 (métrica principal)",
               "test_g1": "Prueba G1 (misma familia que el entrenamiento)",
               "val": "Validación (solo para elegir hiperparámetros; optimista)"}
    for prueba in ("test_g2", "test_g1", "val"):
        lineas += tabla_md(df[df.prueba == prueba], titulos[prueba])
    print("Validación cruzada de 5 particiones (G1, semilla 42)...")
    cv = validacion_cruzada(42, 5)
    cv.to_csv(EXP / "cv5_seed42.csv", index=False)
    cv.groupby("modelo", sort=False)[CLAVES].agg(["mean", "std"]).to_csv(EXP / "resumen_cv5.csv")
    lineas += tabla_md(cv, "Validación cruzada de 5 particiones por conversación (G1, semilla 42; media ± desv. entre particiones)")
    (EXP / "resumen.md").write_text("\n".join(lineas), encoding="utf-8")
    figura(df[df.prueba == "test_g2"])
    print("\n".join(lineas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    main(ap.parse_args().seeds)
