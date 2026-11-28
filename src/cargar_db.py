"""Carga los CSV generados en PostgreSQL (o en cualquier base compatible con SQLAlchemy).

Uso:  docker compose up -d                      # levanta PostgreSQL 16
      python -m src.cargar_db --seed 42
La URL se toma de la variable ALBERT_DB_URL; por defecto apunta al contenedor:
      postgresql+psycopg2://albert:albert@localhost:5432/albert_sim
"""
import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .generar_datos import DATA

URL = os.environ.get("ALBERT_DB_URL", "postgresql+psycopg2://albert:albert@localhost:5432/albert_sim")
ESQUEMA = Path(__file__).resolve().parent / "sql" / "esquema.sql"


def cargar(seed: int, url: str = URL) -> dict:
    d = DATA / f"seed_{seed}"
    convs, turnos = pd.read_csv(d / "conversaciones.csv"), pd.read_csv(d / "turnos.csv")
    convs = convs.assign(fuente="sintetico", seed=seed,
                         contacto_hash=convs.conversacion_id.map(lambda x: hashlib.sha256(f"sintetico:{x}".encode()).hexdigest()))
    eng = create_engine(url)
    with eng.begin() as cx:
        sql = ESQUEMA.read_text(encoding="utf-8")
        if eng.dialect.name != "postgresql":           # SQLite para pruebas locales
            sql = sql.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT").replace("CHAR(64)", "TEXT")
        for stmt in [s for s in sql.split(";") if s.strip()]:
            cx.execute(text(stmt))
        cx.execute(text("DELETE FROM mensaje WHERE conversacion_id IN (SELECT conversacion_id FROM conversacion WHERE seed = :s)"), {"s": seed})
        cx.execute(text("DELETE FROM conversacion WHERE seed = :s"), {"s": seed})
        cols_c = ["conversacion_id", "contacto_hash", "fuente", "seed", "variante", "particion", "intencion", "tipo_usuario",
                  "escritura", "usa_voz", "caso_complejo", "requiere_asesor", "turno_oraculo", "n_turnos"]
        convs[cols_c].astype(object).where(convs[cols_c].notna(), None).to_sql("conversacion", cx, if_exists="append", index=False)
        cols_m = ["conversacion_id", "turno", "texto", "es_voz", "obs", "s_pide_asesor", "s_negativo", "s_repite", "s_voz",
                  "s_sin_atajo", "s_cierre", "s_atajo", "estado_latente", "frustracion"]
        turnos[cols_m].to_sql("mensaje", cx, if_exists="append", index=False, chunksize=2000)
        n = cx.execute(text("SELECT particion, COUNT(*), ROUND(AVG(requiere_asesor)*100, 1) FROM conversacion "
                            "WHERE seed = :s GROUP BY particion ORDER BY particion"), {"s": seed}).fetchall()
    return {p: {"conversaciones": c, "pct_requiere_asesor": float(r)} for p, c, r in n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    for p, v in cargar(a.seed, a.url).items():
        print(f"{p:8s} {v['conversaciones']:5d} conversaciones · {v['pct_requiere_asesor']:.1f} % requiere asesor")
