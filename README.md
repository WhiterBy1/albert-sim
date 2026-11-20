# Albert-sim · Escalamiento a asesor humano bajo incertidumbre

Proyecto final de **Inteligencia Artificial (ISCO-A06A), UTB** · Ruta B: razonamiento con incertidumbre.
Autor: José David Quiñonez Rehenals (T00083600) · Docente: Edwin Puertas.

«Albert» es el agente de WhatsApp de la Unidad de Patología Clínica (UPC). Hoy decide
cuándo pasar una conversación a un asesor con **reglas fijas** (palabra «asesor», enojo
o 3 fallos seguidos). Este repositorio construye, **fuera de producción y sin enviar datos
a ninguna API**, un módulo que estima `P(requiere asesor | observaciones 1..t)` con un
**modelo oculto de Markov (HMM) y filtrado hacia adelante**, y lo compara con una **regresión logística**
sin estado, con las reglas actuales y con la clase mayoritaria.

## Ejecución (3 comandos; solo se necesita Docker)

```bash
git clone <URL-del-repositorio> albert-sim && cd albert-sim
docker compose up --build        # primera vez ≈ 3 min: genera datos, carga PostgreSQL 16 y abre la app
# abrir http://localhost:8501
```

Opcionales, con la app arriba (en otra terminal, dentro de la carpeta):

```bash
docker compose exec app python -m src.run_experimentos   # reproduce todas las tablas en experiments/ (≈ 3 min)
docker compose exec app python -m pytest -q tests        # 9 pruebas
```

Para apagar: `docker compose down` (con `-v` también borra la base). Sin Docker: Python 3.12+,
`pip install -r requirements.txt`, `python -m src.run_experimentos` y `streamlit run src/app.py`
(sin PostgreSQL la app lee los CSV que genera el script).

### La aplicación (prototipo funcional)

| Pestaña | Qué hace |
|---|---|
| **Chat en vivo** | Escribes como un paciente (con casilla de nota de voz / familiar). Muestra, turno a turno, la decisión de las reglas actuales, del HMM y de la logística, con su probabilidad. |
| **Conversaciones de la base** | Recorre las conversaciones guardadas en PostgreSQL y dibuja las curvas de probabilidad contra la etiqueta real. |
| **Resultados** | Tablas de los experimentos, validación cruzada, errores por tipo y la muestra de errores. |

### Prueba real con el piloto (solo dentro de la UPC)

```bash
export ALBERT_SAL='<secreto fuera del repositorio>'
python -m src.anonimizar_piloto export_panel.csv   # anonimiza y genera revision_manual.txt
python -m src.evaluar_piloto --seed 42             # solo evalúa; escribe métricas agregadas
```

Los datos del piloto (entrada y salida) están en `.gitignore` y **nunca** se suben.

## Estructura

```
Dockerfile, docker-compose.yml   PostgreSQL 16 + app Streamlit (un solo `docker compose up`)
src/
  arrancar.py           arranque del contenedor: genera datos, carga la base y lanza la app
  app.py                prototipo Streamlit (chat en vivo, conversaciones, resultados)
  texto.py              normalización y simulación de mala escritura
  observaciones.py      sensor: palabras clave de Albert -> símbolo discreto por turno
  generar_datos.py      generador sintético G1 (train/val/test) y G2 (prueba independiente)
  cargar_db.py, sql/    esquema reducido del panel y carga en PostgreSQL
  modelos.py            reglas fijas, clase mayoritaria, HMM con filtrado y regresión logística con ventana
  metricas.py           métricas predeclaradas (recall, F1, innecesarios, retraso, ECE, subgrupos, Wilson)
  run_experimentos.py   5 semillas + validación cruzada de 5 particiones -> experiments/
  analisis_errores.py   clasifica los errores por tipo, con hipótesis de causa -> experiments/
  anonimizar_piloto.py  anonimización del export del piloto (correo, teléfono, documento, orden, dirección, nombres)
  evaluar_piloto.py     evaluación en la prueba real (n = 120), con intervalos de Wilson
notebooks/01_exploracion.ipynb   estadísticas descriptivas, calibración, comparación de modelos, errores
experiments/            resultados que respaldan cada tabla del informe (semillas 42–46)
tests/                  9 pruebas automáticas (+ un export de ejemplo 100 % ficticio)
docs/bitacora_ia.md     bitácora humano-IA
docs/hitos/             PDF de los Hitos 1 y 2
```

## Datos

* **Sintéticos** (licencia CC BY 4.0): 2 000 conversaciones G1 por semilla (70 % train, 15 % val, 15 % test) y
  300 conversaciones G2 generadas con otras plantillas, más ruido y otra semilla. Partición **por conversación**.
  Se calibran solo con conteos agregados del piloto: semilla 42 → 33,8 % requiere asesor (piloto: 36 %) y
  23,2 % de mensajes al LLM (piloto: 23 %). Detalle en `data/README.md`.
* **Piloto de Albert** (31 días, 120 conversaciones): anonimizado, solo para la prueba real; nunca para entrenar.

## Resultados (Hito 2) · prueba independiente G2, 5 semillas (media ± desv.)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | Brecha F1 |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0,890 ± 0,030 | 0,819 ± 0,026 | 0,758 ± 0,026 | 24,2 ± 2,6 | 1,0 | — | 0,079 ± 0,031 |
| Línea base 2 · Clase mayoritaria | 0,000 | 0,000 | — | — | — | 0,016 ± 0,018 | 0,000 |
| (A) HMM, τ* = 0,20 (predeclarado) | 1,000 ± 0,000 | 0,708 ± 0,013 | 0,548 ± 0,015 | 45,2 ± 1,5 | 0,0 | 0,120 ± 0,007 | 0,119 ± 0,043 |
| (A') HMM, τ elegido en validación | 0,939 ± 0,058 | 0,914 ± 0,027 | 0,895 ± 0,065 | 10,5 ± 6,5 | 0,8 | 0,132 ± 0,025 | 0,133 ± 0,050 |
| (B) Logística, τ* = 0,20 | 0,998 ± 0,005 | 0,797 ± 0,032 | 0,665 ± 0,043 | 33,5 ± 4,3 | 0,0 | 0,149 ± 0,005 | 0,145 ± 0,028 |
| (B') Logística, τ elegido en validación | 0,982 ± 0,013 | 0,919 ± 0,007 | 0,863 ± 0,009 | 13,7 ± 0,9 | 0,0 | 0,112 ± 0,009 | 0,160 ± 0,023 |

Umbrales de éxito declarados en el Hito 1: Recall ≥ 0,90 · F1 ≥ 0,80 y ≥ +5 pp sobre las reglas · innecesarios ≤ 20 % · retraso ≤ 1 turno y menor que las reglas · ECE ≤ 0,10 · brecha F1 por subgrupo ≤ 0,10.

**Lectura honesta**

* Con el umbral **predeclarado** (τ* = 0,20, que solo podía bajar), el HMM y la logística alcanzan recall ≈ 1,0 pero con 34–45 % de escalamientos innecesarios: **no cumplen** F1 ni innecesarios.
* Elegir τ en validación (máx. F1 con recall ≥ 0,90) es un **desvío del Hito 1** y se informa por separado; con él, A' y B' cumplen recall, F1 (+9 pp sobre las reglas), innecesarios y retraso. **Ningún modelo cumple ECE ≤ 0,10 ni brecha por subgrupo ≤ 0,10** (el ECE se mide por conversación sobre la probabilidad máxima; es una medida exigente).
* **El HMM no supera a la regresión logística con ventana** (F1 0,914 vs 0,919, dentro del ruido; la logística tiene menos retraso y menor desviación). La ventaja temporal del HMM no se observa en estos datos; el generador acumula evidencia de forma que una ventana de turnos la captura.
* Validación cruzada de 5 particiones (G1, semilla 42): A' F1 = 0,949 ± 0,006; B' F1 = 0,966 ± 0,008; reglas 0,815 ± 0,028 (`experiments/resumen.md`).
* Los datos son sintéticos: estas cifras miden el método, no el desempeño en producción. La prueba real (n = 120) queda para el Hito 3.
* Errores (semilla 42, G2): el HMM A' falla en 9 conversaciones que sí requerían asesor (6 con notas de voz) y en 3 que no; 16 casos clasificados con hipótesis por validar en `experiments/errores_muestra.csv`.

Todo está en `experiments/` (`resumen.md`, `resultados_por_semilla.csv`, `seed_<n>.json`, `cv5_seed42.csv`, `subgrupos_test_g2.csv`, `errores_*.csv`, figuras).

## Licencia

Código: MIT (`LICENSE`). Datos sintéticos: CC BY 4.0. Datos del piloto: propiedad de la UPC, no incluidos.
