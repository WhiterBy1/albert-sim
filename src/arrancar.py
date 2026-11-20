"""Arranque del contenedor: prepara todo y lanza la app. Un solo paso para quien clona el repositorio.

1. Genera los datos sintéticos (semilla 42) si no existen.
2. Espera a PostgreSQL y carga el esquema y las conversaciones (si la base está vacía).
3. Lanza la aplicación Streamlit en el puerto 8501.

Uso (dentro del contenedor, lo hace docker compose):  python -m src.arrancar
"""
import os
import subprocess
import sys
import time

from sqlalchemy import create_engine, text

from .cargar_db import URL, cargar
from .generar_datos import DATA, construir

SEED = 42


def esperar_db(url: str, intentos: int = 40):
    for i in range(intentos):
        try:
            with create_engine(url).connect() as cx:
                cx.execute(text("SELECT 1"))
            return
        except Exception:
            print(f"Esperando a PostgreSQL ({i + 1}/{intentos})...", flush=True)
            time.sleep(2)
    raise SystemExit("PostgreSQL no respondió.")


def main():
    if not (DATA / f"seed_{SEED}" / "conversaciones.csv").exists():
        print("Generando datos sintéticos (semilla 42)...", flush=True)
        construir(SEED)
    esperar_db(URL)
    eng = create_engine(URL)
    with eng.connect() as cx:
        hay = eng.dialect.has_table(cx, "conversacion") and cx.execute(text("SELECT COUNT(*) FROM conversacion")).scalar() > 0
    if not hay:
        print("Cargando conversaciones en PostgreSQL...", flush=True)
        for p, v in cargar(SEED).items():
            print(f"  {p}: {v['conversaciones']} conversaciones", flush=True)
    print("Abriendo la aplicación en http://localhost:8501", flush=True)
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port=8501",
                              "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"],
                             env=os.environ))


if __name__ == "__main__":
    main()
