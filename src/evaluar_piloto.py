"""Evalúa los decisores en la prueba real (piloto anonimizado). Solo evalúa: nunca entrena.

Los modelos se ajustan con los datos sintéticos de entrenamiento (G1) de la
semilla indicada. Al repositorio solo se escriben métricas agregadas con
intervalo de Wilson al 95 %: experiments/piloto_resultados.json

Uso:  python -m src.evaluar_piloto --seed 42
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .anonimizar_piloto import SALIDA
from .metricas import evaluar, wilson
from .modelos import HMMEscalamiento, Mayoritaria, ReglasFijas, aplicar, ajustar_tau
from .observaciones import observar
from .run_experimentos import EXP, cargar

TERCERO = re.compile(r"\bmi (mam[aá]|pap[aá]|hijo|hija|esposo|esposa|abuel[oa]|herman[oa]|t[ií][oa])\b|\bfamiliar\b", re.I)


def preparar(convs: pd.DataFrame, turnos: pd.DataFrame):
    obs = []
    for cid, g in turnos.groupby("conversacion_id", sort=False):
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
    return convs, turnos


def main(seed):
    archivo = SALIDA / "piloto_anonimizado.csv"
    if not archivo.exists():
        raise SystemExit("No hay piloto anonimizado. Ejecute antes: python -m src.anonimizar_piloto <export.csv>")
    convs, turnos = preparar(pd.read_csv(SALIDA / "piloto_conversaciones.csv"), pd.read_csv(archivo))
    c, t = cargar(seed)
    tr = (c[c.particion == "train"], t[t.conversacion_id.isin(c[c.particion == "train"].conversacion_id)])
    va = (c[c.particion == "val"], t[t.conversacion_id.isin(c[c.particion == "val"].conversacion_id)])
    hmm = HMMEscalamiento().ajustar(*tr)
    ajustar_tau(hmm, *va)
    salida = {"n_conversaciones": int(len(convs)), "pct_con_asesor": 100 * float(convs.requiere_asesor.mean()),
              "seed_entrenamiento": seed, "tau_hmm": hmm.tau, "modelos": {}}
    for m in (ReglasFijas(), Mayoritaria().ajustar(*tr), hmm):
        r = evaluar(aplicar(m, convs, turnos))
        r["recall_ic95"] = wilson(r["tp"], r["tp"] + r["fn"])
        r["precision_ic95"] = wilson(r["tp"], r["tp"] + r["fp"])
        salida["modelos"][m.nombre] = r
    EXP.mkdir(exist_ok=True)
    (EXP / "piloto_resultados.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=float))
    for k, r in salida["modelos"].items():
        print(f"{k:42s} recall {r['recall']:.3f} {tuple(round(x, 3) for x in r['recall_ic95'])}  F1 {r['f1']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
