"""Métricas predeclaradas en el Hito 1 (Tabla de métricas y umbrales de éxito)."""
import numpy as np
import pandas as pd

UMBRALES = {"recall": 0.90, "f1": 0.80, "pct_innecesarios": 20.0, "retraso_mediana": 1.0,
            "ece": 0.10, "brecha_f1_subgrupo": 0.10}
SUBGRUPOS = ["tipo_usuario", "escritura", "usa_voz"]


def _prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def ece(y, score, bins=10):
    """Expected Calibration Error con bins de igual ancho."""
    y, score = np.asarray(y, float), np.asarray(score, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - score[m].mean())
    return float(total)


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporción (se usa con el piloto, n = 120)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(c - h), float(c + h))


def evaluar(res: pd.DataFrame) -> dict:
    """`res` tiene una fila por conversación con: requiere_asesor, escala (0/1),
    turno_escala, turno_oraculo, score (prob. o NaN) y columnas de subgrupo."""
    y, p = res.requiere_asesor.values.astype(int), res.escala.values.astype(int)
    tp, fp, fn, prec, rec, f1 = _prf(y, p)
    m = (y == 1) & (p == 1)
    retraso = np.maximum(0, res.loc[m, "turno_escala"].values - res.loc[m, "turno_oraculo"].values)
    out = {"n": int(len(res)), "positivos": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
           "precision": prec, "recall": rec, "f1": f1,
           "exactitud": float((y == p).mean()),
           "pct_innecesarios": 100 * fp / (tp + fp) if tp + fp else 0.0,
           "retraso_mediana": float(np.median(retraso)) if len(retraso) else float("nan"),
           "retraso_medio": float(np.mean(retraso)) if len(retraso) else float("nan"),
           "ece": ece(y, res.score.values) if res.score.notna().all() else float("nan")}
    brechas = {}
    for g in SUBGRUPOS:
        f1s = {str(k): _prf(d.requiere_asesor.values, d.escala.values)[5] for k, d in res.groupby(g)}
        out[f"f1_por_{g}"] = f1s
        brechas[g] = max(f1s.values()) - min(f1s.values()) if len(f1s) > 1 else 0.0
    out["brecha_f1_max"] = max(brechas.values())
    out["brechas_f1"] = brechas
    return out
