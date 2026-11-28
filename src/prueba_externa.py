"""Prueba externa G3: conversaciones de OTRO generador (IPS X), incluidas 1 000 generadas con dos agentes LLM.

Motivación: G1 y G2 salen del mismo autor y comparten supuestos, de modo que los modelos podrían estar
aprendiendo el generador. G3 se construyó aparte (data/externo/README.md):
  * G3-guion   (500): conversaciones guionadas con plantillas y ruido de escritura.
  * G3-agentes (500): simulación de agentes (usuario con objetivos y paciencia vs. bot de reglas).
  * G3-llm   (1 000): usuario y bot simulados por un modelo de lenguaje (Claude Haiku) con respuestas libres.
Los modelos NO se reentrenan: se ajustan con G1 (entrenamiento) y eligen umbrales con G1 (validación) de cada
semilla, igual que en run_experimentos. G3 solo se evalúa.

Etiqueta de G3: requiere asesor si el simulador marcó un momento en que el bot ya no podía resolver
(objetivo que solo un humano atiende o fallos repetidos en un objetivo importante) o si el usuario pidió un
humano; turno oráculo = primero de esos momentos (en turnos de usuario). Se informa además el proxy
«terminó con asesor» para mostrar cuánto cambia la conclusión con una etiqueta sesgada.

Uso:  python -m src.prueba_externa            (≈ 1–2 min)
Salida: experiments/externa_resumen.md, externa_por_semilla.csv, externa_subgrupos.csv,
        externa_errores_por_tipo.csv, externa_errores_muestra.csv, fig_externa_g3.png
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .analisis_errores import clasificar
from .metricas import evaluar, wilson
from .modelos import aplicar
from .observaciones import observar
from .run_experimentos import EXP, N_HMM_V, N_LR_V, N_REGLAS, ORDEN, cargar, entrenar_modelos, sub

DIR = Path(__file__).resolve().parent.parent / "data" / "externo"
SEEDS = [42, 43, 44, 45, 46]
TERCERO = re.compile(r"\bmi (mam[aá]|pap[aá]|hijo|hija|esposo|esposa|abuel[oa]|herman[oa]|t[ií][oa]|nieto|nieta)\b|"
                     r"\bde mi (mam[aá]|pap[aá]|hijo|hija|esposo|esposa|abuel[oa])\b|\bfamiliar\b", re.I)
GRUPOS = {"G3-llm": ["G3-llm"], "G3-guion+agentes": ["G3-guion", "G3-agentes"], "G3 total": ["G3-llm", "G3-guion", "G3-agentes"]}
CLAVES = ["recall", "f1", "precision", "pct_innecesarios", "retraso_mediana", "ece"]


def cargar_externa():
    convs = pd.read_csv(DIR / "conversaciones.csv")
    turnos = pd.read_csv(DIR / "turnos.csv", keep_default_na=False)
    obs = []
    for _, g in turnos.groupby("conversacion_id", sort=False):
        previo = None
        for _, r in g.iterrows():
            obs.append(observar(r.texto, bool(r.es_voz), previo)["obs"])
            previo = None if r.es_voz else r.texto
    turnos = turnos.assign(obs=obs)
    texto = turnos.groupby("conversacion_id").texto.apply(" ".join)
    convs = convs.assign(tipo_usuario=convs.conversacion_id.map(lambda c: "tercero" if TERCERO.search(texto.get(c, "")) else "titular"),
                         escritura="no_medida")
    convs["turno_oraculo"] = convs.turno_oraculo.astype("float")
    return convs, turnos


def bootstrap_f1(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, pa, pb = a.requiere_asesor.values, a.escala.values, b.escala.values

    def f1(yy, pp):
        tp = ((yy == 1) & (pp == 1)).sum(); fp = ((yy == 0) & (pp == 1)).sum(); fn = ((yy == 1) & (pp == 0)).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1(y[i], pa[i]) - f1(y[i], pb[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    convs, turnos = cargar_externa()
    filas, subg, errores, muestra, delta = [], [], [], [], {}
    for seed in SEEDS:
        c, t = cargar(seed)
        modelos = entrenar_modelos(sub(c, t, "train"), sub(c, t, "val"))
        for grupo, vars_ in GRUPOS.items():
            cg = convs[convs.variante.isin(vars_)]
            tg = turnos[turnos.conversacion_id.isin(cg.conversacion_id)]
            res = {m.nombre: aplicar(m, cg, tg) for m in modelos}
            for nombre, r in res.items():
                for etiqueta in ("simulador", "proxy_termino_con_asesor"):
                    rr = r if etiqueta == "simulador" else r.assign(requiere_asesor=r.termino_con_asesor)
                    ev = evaluar(rr)
                    filas.append({"seed": seed, "grupo": grupo, "etiqueta": etiqueta, "modelo": nombre, "n": ev["n"],
                                  "positivos": ev["positivos"], "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"],
                                  **{k: ev[k] for k in CLAVES}})
                if grupo == "G3 total":
                    ev = evaluar(r)
                    for g in ("tipo_usuario", "usa_voz"):
                        for k, v in ev[f"f1_por_{g}"].items():
                            subg.append({"seed": seed, "modelo": nombre, "subgrupo": g, "valor": k, "f1": v})
                    from .metricas import _prf
                    for k, d in r.groupby("perfil"):
                        subg.append({"seed": seed, "modelo": nombre, "subgrupo": "perfil", "valor": k,
                                     "f1": _prf(d.requiere_asesor.values, d.escala.values)[5]})
            if seed == 42:
                for nombre in res:
                    if nombre != N_REGLAS and "mayoritaria" not in nombre.lower():
                        delta[(grupo, nombre)] = bootstrap_f1(res[nombre], res[N_REGLAS])
                if grupo == "G3 total":
                    por_conv = dict(tuple(tg.groupby("conversacion_id", sort=False)))
                    for nombre in (N_REGLAS, N_HMM_V, N_LR_V):
                        r = res[nombre]
                        for _, f in r[r.requiere_asesor != r.escala].iterrows():
                            sim = por_conv[f.conversacion_id].obs.tolist()
                            tipo = clasificar(f, sim)
                            errores.append({"modelo": nombre, "variante": f.variante, "error": tipo.split("_")[0], "tipo_error": tipo})
                            if nombre == N_HMM_V:
                                muestra.append({"conversacion_id": f.conversacion_id, "variante": f.variante, "tipo_error": tipo,
                                                "turno_oraculo": f.turno_oraculo, "turno_escala": f.turno_escala,
                                                "simbolos": " ".join(sim),
                                                "mensajes": " || ".join(por_conv[f.conversacion_id].texto.tolist())[:600]})

    df = pd.DataFrame(filas)
    EXP.mkdir(exist_ok=True)
    df.to_csv(EXP / "externa_por_semilla.csv", index=False)
    pd.DataFrame(subg).groupby(["modelo", "subgrupo", "valor"], sort=False).f1.agg(["mean", "std"]).round(3).reset_index() \
        .to_csv(EXP / "externa_subgrupos.csv", index=False)
    pd.DataFrame(errores).value_counts().rename("casos").reset_index().to_csv(EXP / "externa_errores_por_tipo.csv", index=False)
    m = pd.DataFrame(muestra)
    m = m.groupby("tipo_error", group_keys=False).apply(lambda d: d.sample(min(len(d), 4), random_state=0))
    m.to_csv(EXP / "externa_errores_muestra.csv", index=False)

    lin = ["# Prueba externa G3 (generador independiente; los modelos se entrenan solo con G1)", "",
           "Media ± desv. en 5 semillas de entrenamiento (42–46). ΔF1 vs reglas: bootstrap pareado (2 000 remuestreos, semilla 42).", ""]
    for etiqueta in ("simulador", "proxy_termino_con_asesor"):
        for grupo in GRUPOS:
            d = df[(df.grupo == grupo) & (df.etiqueta == etiqueta)]
            n, pos = int(d.n.iloc[0]), int(d.positivos.iloc[0])
            lin += [f"## {grupo} · etiqueta: {etiqueta} (n = {n}; {100 * pos / n:.1f} % requiere asesor)", "",
                    "| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |",
                    "|---|---|---|---|---|---|---|---|"]
            for nombre in ORDEN:
                x = d[d.modelo == nombre]
                f = lambda k, dec=3: f"{x[k].mean():.{dec}f} ± {x[k].std():.{dec}f}" if x[k].notna().any() else "—"
                dd = delta.get((grupo, nombre)) if etiqueta == "simulador" else None
                ds = f"{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]" if dd else "—"
                lin.append(f"| {nombre} | {f('recall')} | {f('f1')} | {f('precision')} | {f('pct_innecesarios', 1)} | "
                           f"{f('retraso_mediana', 1)} | {f('ece')} | {ds} |")
            lin.append("")
    figura(df)
    (EXP / "externa_resumen.md").write_text("\n".join(lin) + "\n", encoding="utf-8")
    print("\n".join(lin))


def figura(df):
    """F1 de reglas, A' y B' en G2 (mismo generador) frente a G3 (generadores externos); barras = desv. entre semillas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g2 = pd.read_csv(EXP / "resultados_por_semilla.csv")
    g2 = g2[g2.prueba == "test_g2"]
    sim = df[df.etiqueta == "simulador"]
    pruebas = [("G2 (mismo generador)", g2, None), ("G3-guion+agentes", sim, "G3-guion+agentes"), ("G3-llm", sim, "G3-llm")]
    modelos = [(N_REGLAS, "Reglas (Albert)", "#7f7f7f"), (N_HMM_V, "(A') HMM", "#1f77b4"), (N_LR_V, "(B') Logística", "#ff7f0e")]
    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
    x = np.arange(len(pruebas)); w = 0.26
    for j, (m, etiqueta, color) in enumerate(modelos):
        med, sd = [], []
        for _, d, grupo in pruebas:
            d = d[d.modelo == m] if grupo is None else d[(d.modelo == m) & (d.grupo == grupo)]
            med.append(d.f1.mean()); sd.append(d.f1.std())
        ax.bar(x + (j - 1) * w, med, w, yerr=sd, capsize=2, label=etiqueta, color=color)
        for xi, v, e in zip(x + (j - 1) * w, med, sd):
            ax.text(xi, v + (e if e == e else 0) + 0.015, f"{v:.2f}".replace(".", ","), ha="center", fontsize=6)
    ax.set_xticks(x, [p[0] for p in pruebas], fontsize=7)
    ax.set_ylabel("F1", fontsize=8); ax.set_ylim(0, 1.0); ax.tick_params(labelsize=7)
    ax.axhline(0.80, ls="--", lw=0.8, color="k"); ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(EXP / "fig_externa_g3.png"); plt.close(fig)


if __name__ == "__main__":
    main()
