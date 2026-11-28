"""Evalúa los decisores en la prueba real (piloto anonimizado). Solo evalúa: nunca entrena.

Los modelos se ajustan con los datos sintéticos de entrenamiento (G1) de la semilla
indicada y sus umbrales se eligen en la validación sintética, igual que en
`run_experimentos`. Al repositorio solo se escriben métricas AGREGADAS:
    experiments/piloto_resultados.json   (todas las métricas + IC de Wilson y bootstrap)
    experiments/piloto_resumen.md        (tabla lista para el informe)
    experiments/piloto_errores_por_tipo.csv (solo conteos por tipo, sin textos)

Etiqueta:
  * Si existe data/piloto/etiquetas_manual.csv (conversacion_id, requiere_asesor,
    turno_oraculo), se usa esa etiqueta revisada a mano, ciega a los modelos.
  * Si no, se usa el proxy «terminó con asesor», que favorece a las reglas porque
    en producción fueron las reglas las que decidieron escalar (se declara).

Uso:  python -m src.evaluar_piloto --seed 42
      python -m src.evaluar_piloto --plantilla-etiquetas   # crea la plantilla para etiquetar a mano
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

from .analisis_errores import clasificar
from .anonimizar_piloto import SALIDA
from .metricas import evaluar, wilson
from .modelos import aplicar
from .observaciones import observar
from .run_experimentos import EXP, cargar, entrenar_modelos, sub

TERCERO = re.compile(r"\bmi (mam[aá]|pap[aá]|hijo|hija|esposo|esposa|abuel[oa]|herman[oa]|t[ií][oa])\b|\bfamiliar\b", re.I)
ETIQUETAS = SALIDA / "etiquetas_manual.csv"


def preparar(convs: pd.DataFrame, turnos: pd.DataFrame):
    obs = []
    for _, g in turnos.groupby("conversacion_id", sort=False):
        previo = None
        for _, r in g.iterrows():
            obs.append(observar(r.texto, bool(r.es_voz), previo)["obs"])
            previo = None if r.es_voz else r.texto
    turnos = turnos.assign(obs=obs)
    texto_conv = turnos.groupby("conversacion_id").texto.apply(" ".join)
    convs = convs.assign(
        tipo_usuario=convs.conversacion_id.map(lambda c: "tercero" if TERCERO.search(texto_conv.get(c, "")) else "titular"),
        usa_voz=convs.conversacion_id.map(turnos.groupby("conversacion_id").es_voz.max()).fillna(0).astype(int),
        escritura="no_medida")
    fuente = "proxy_termino_con_asesor"
    if ETIQUETAS.exists():
        et = pd.read_csv(ETIQUETAS)
        if et.requiere_asesor.isna().any():
            raise SystemExit(f"{ETIQUETAS} tiene filas sin etiquetar. Complételas o borre el archivo para usar el proxy «terminó con asesor».")
        convs = convs.drop(columns=["requiere_asesor", "turno_oraculo"]).merge(
            et[["conversacion_id", "requiere_asesor", "turno_oraculo"]], on="conversacion_id", how="inner")
        fuente = "manual_ciega"
    return convs, turnos, fuente


def bootstrap_f1(res_a, res_b, n=2000, seed=0):
    """IC 95 % de F1(a) - F1(b) remuestreando conversaciones (pareado)."""
    rng = np.random.default_rng(seed)
    y = res_a.requiere_asesor.values
    pa, pb = res_a.escala.values, res_b.escala.values

    def f1(yy, pp):
        tp = ((yy == 1) & (pp == 1)).sum(); fp = ((yy == 0) & (pp == 1)).sum(); fn = ((yy == 1) & (pp == 0)).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    difs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        difs.append(f1(y[i], pa[i]) - f1(y[i], pb[i]))
    return float(np.mean(difs)), (float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5)))


def plantilla():
    t = pd.read_csv(SALIDA / "piloto_anonimizado.csv")
    p = pd.DataFrame({"conversacion_id": t.conversacion_id.unique()})
    p["requiere_asesor"], p["turno_oraculo"], p["nota"] = "", "", ""
    p.to_csv(ETIQUETAS, index=False)
    print(f"Plantilla creada: {ETIQUETAS}. Etiquete leyendo revision_manual.txt SIN mirar las salidas de los modelos.")


def main(seed):
    if not (SALIDA / "piloto_anonimizado.csv").exists():
        raise SystemExit("No hay piloto anonimizado. Ejecute antes: python -m src.anonimizar_piloto <export.csv>")
    convs, turnos, fuente = preparar(pd.read_csv(SALIDA / "piloto_conversaciones.csv"),
                                     pd.read_csv(SALIDA / "piloto_anonimizado.csv"))
    c, t = cargar(seed)
    modelos = entrenar_modelos(sub(c, t, "train"), sub(c, t, "val"))
    salida = {"n_conversaciones": int(len(convs)), "n_positivos": int(convs.requiere_asesor.sum()),
              "pct_requiere_asesor": 100 * float(convs.requiere_asesor.mean()), "fuente_etiqueta": fuente,
              "seed_entrenamiento": seed, "pct_turnos_por_simbolo": (100 * turnos.obs.value_counts(normalize=True)).round(1).to_dict(),
              "pct_mensajes_al_llm_sensor": 100 * float(turnos.obs.isin(["SIN_ATAJO", "VOZ", "REPITE"]).mean()),
              "modelos": {}, "comparaciones_f1_vs_reglas": {}}
    resultados, errores = {}, []
    por_conv = dict(tuple(turnos.groupby("conversacion_id", sort=False)))
    for m in modelos:
        res = aplicar(m, convs, turnos)
        resultados[m.nombre] = res
        r = evaluar(res)
        r["tau"] = getattr(m, "tau", None)
        r["recall_ic95"] = wilson(r["tp"], r["tp"] + r["fn"])
        r["precision_ic95"] = wilson(r["tp"], r["tp"] + r["fp"])
        salida["modelos"][m.nombre] = r
        for _, f in res[res.requiere_asesor != res.escala].iterrows():
            errores.append({"modelo": m.nombre, "error": "FN" if f.requiere_asesor else "FP",
                            "tipo_error": clasificar(f, por_conv[f.conversacion_id].obs.tolist())})
    reglas = next(k for k in resultados if "Reglas" in k)
    for k, res in resultados.items():
        if k != reglas and "mayoritaria" not in k.lower():
            media, ic = bootstrap_f1(res, resultados[reglas])
            salida["comparaciones_f1_vs_reglas"][k] = {"dif_media": media, "ic95": ic}
    EXP.mkdir(exist_ok=True)
    (EXP / "piloto_resultados.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    if errores:
        pd.DataFrame(errores).value_counts().rename("casos").reset_index().to_csv(EXP / "piloto_errores_por_tipo.csv", index=False)
    lineas = [f"# Prueba real · piloto de Albert (n = {salida['n_conversaciones']}, etiqueta: {fuente})", "",
              "| Modelo | Recall [IC95] | Precisión [IC95] | F1 | % innecesarios | Retraso (mediana) | ΔF1 vs reglas [IC95] |",
              "|---|---|---|---|---|---|---|"]
    for k, r in salida["modelos"].items():
        cmp_ = salida["comparaciones_f1_vs_reglas"].get(k)
        d = f"{cmp_['dif_media']:+.3f} [{cmp_['ic95'][0]:+.3f}, {cmp_['ic95'][1]:+.3f}]" if cmp_ else "—"
        lineas.append(f"| {k} | {r['recall']:.3f} [{r['recall_ic95'][0]:.2f}, {r['recall_ic95'][1]:.2f}] | "
                      f"{r['precision']:.3f} [{r['precision_ic95'][0]:.2f}, {r['precision_ic95'][1]:.2f}] | {r['f1']:.3f} | "
                      f"{r['pct_innecesarios']:.1f} | {r['retraso_mediana']} | {d} |")
    (EXP / "piloto_resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plantilla-etiquetas", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):          # consola de Windows (cp1252) no puede imprimir «Δ» ni «τ»
        sys.stdout.reconfigure(errors="replace")
    plantilla() if a.plantilla_etiquetas else main(a.seed)
