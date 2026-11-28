# Albert-sim · Escalamiento a asesor humano bajo incertidumbre

Proyecto final de **Inteligencia Artificial (ISCO-A06A), UTB** · Ruta B: razonamiento con incertidumbre.
Autor: José David Quiñonez Rehenals (T00083600) · Docente: Edwin Puertas.

«Albert» es el agente de WhatsApp de la Unidad de Patología Clínica (UPC). Hoy decide
cuándo pasar una conversación a un asesor con **reglas fijas** (palabra «asesor», enojo
o 3 fallos seguidos). Este repositorio construye, **fuera de producción y sin enviar datos
a ninguna API**, un módulo que estima `P(requiere asesor | observaciones 1..t)` con un
**modelo oculto de Markov (HMM) y filtrado hacia adelante**, y lo compara con una **regresión logística**
sin estado, con las reglas actuales y con la clase mayoritaria.

## Ejecución directa en Windows (un clic, sin Docker)

1. Instala Python 3.12 o posterior desde [python.org](https://www.python.org/downloads/).
2. Haz doble clic en `Iniciar-Albert-Sim.bat`.
3. Espera a que se abra la aplicación en el navegador.

La primera vez puede tardar unos minutos mientras se prepara el entorno y se instalan las
dependencias. Las siguientes ejecuciones son más rápidas. La aplicación trabaja con datos
locales, sin Docker, PostgreSQL ni conexiones externas.

Para detenerla, cierra la ventana de consola que muestra el servidor.

## Ejecución con Docker (3 comandos)

```bash
git clone <URL-del-repositorio> albert-sim && cd albert-sim
docker compose up --build        # primera vez ≈ 3 min: genera datos, carga PostgreSQL 16 y abre la app
# abrir http://localhost:8501
```

Opcionales, con la app arriba (en otra terminal, dentro de la carpeta):

```bash
docker compose exec app python -m src.run_experimentos   # reproduce todas las tablas en experiments/ (≈ 3 min)
docker compose exec app python -m pytest -q tests        # 10 pruebas
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

Pasos completos en `docs/PRUEBA_REAL_PILOTO.md`. En resumen (con `ALBERT_SAL` definida solo en la terminal):

```bash
python -m src.anonimizar_piloto data/piloto/export_panel.csv   # anonimiza y genera revision_manual.txt
python -m src.evaluar_piloto --seed 42                         # solo evalúa; escribe métricas agregadas
```

El export puede venir con o sin encabezado (columnas `conversacion_id, contacto, fecha_hora, rol, texto[, es_voz]`).
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
  evaluar_piloto.py     evaluación en la prueba real (n = 487): IC de Wilson, bootstrap de ΔF1 y etiqueta manual ciega
  ablacion.py           sensibilidad: reglas k = 2/3/4, ventana de la logística k = 1/3/5, τ del HMM
  prueba_externa.py     prueba externa G3 (2 000 conversaciones de otro generador, 1 000 de ellas con agentes LLM)
notebooks/01_exploracion.ipynb   estadísticas descriptivas, calibración, comparación de modelos, errores
experiments/            resultados que respaldan cada tabla del informe (semillas 42–46)
tests/                  10 pruebas automáticas (+ un export de ejemplo 100 % ficticio)
docs/bitacora_ia.md     bitácora humano-IA
docs/hitos/             PDF de los Hitos 1 y 2
```

## Datos

* **Sintéticos** (licencia CC BY 4.0): 2 000 conversaciones G1 por semilla (70 % train, 15 % val, 15 % test) y
  300 conversaciones G2 generadas con otras plantillas, más ruido y otra semilla. Partición **por conversación**.
  Se calibran solo con conteos agregados del piloto: semilla 42 → 33,8 % requiere asesor (piloto: 36 %) y
  23,2 % de mensajes al LLM (piloto: 23 %). Detalle en `data/README.md`.
* **Prueba externa G3** (CC BY 4.0, `data/externo/`): 2 000 conversaciones de una IPS ficticia generadas
  **con otro generador**: 500 guionadas, 500 con agentes simulados y 1 000 con dos agentes LLM (Claude Haiku).
  Solo se evalúan; los modelos se entrenan con G1. Procedencia, etiqueta y costo en `data/externo/README.md`.
* **Piloto de Albert**: el Hito 1 describía 31 días y 120 conversaciones (sus conteos agregados calibraron G1). La prueba
  real del Hito 3 usa el export del panel del 6 oct. al 27 nov. de 2026 (53 días, 500 conversaciones, 487 evaluadas).
  Anonimizado, solo para la prueba real; nunca para entrenar.

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
* Los datos son sintéticos: estas cifras miden el método, no el desempeño en producción. La prueba real se informa en la sección del Hito 3.
* Errores (semilla 42, G2): el HMM A' falla en 9 conversaciones que sí requerían asesor (6 con notas de voz) y en 3 que no; 16 casos clasificados con hipótesis por validar en `experiments/errores_muestra.csv`.

Todo está en `experiments/` (`resumen.md`, `resultados_por_semilla.csv`, `seed_<n>.json`, `cv5_seed42.csv`, `subgrupos_test_g2.csv`, `errores_*.csv`, figuras).

## Hito 3 · Sensibilidad, prueba externa G3 y prueba real

```bash
python -m src.ablacion          # experiments/ablacion_g2.csv           (≈ 1 min)
python -m src.prueba_externa    # experiments/externa_resumen.md        (≈ 1,5 min)
```

**Sensibilidad (G2, 5 semillas).** Ningún k de las reglas cumple a la vez recall ≥ 0,90 e innecesarios ≤ 20 %
(k = 2: recall 0,983 y 30,9 % innecesarios; k = 4: recall 0,797). La ventana de la logística casi no importa
(F1 0,916–0,921 con k = 1, 3, 5). En el HMM, subir τ de 0,20 a 0,75 baja los innecesarios de 45 % a 5 %
a cambio de 10 puntos de recall.

**Prueba externa G3 (generador independiente; etiqueta del simulador; media ± desv., 5 semillas de entrenamiento).**

| Modelo | G3 total (n = 2 000) F1 | Recall | % innec. | ΔF1 vs reglas [IC95] | G3-llm (n = 1 000) F1 | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|
| Reglas fijas (Albert) | 0,805 | 0,787 | 17,7 | — | 0,857 | — |
| (A') HMM, τ val. | 0,794 ± 0,005 | 0,838 ± 0,038 | 24,4 ± 3,0 | −0,017 [−0,033; −0,001] | 0,845 ± 0,011 | −0,031 [−0,052; −0,011] |
| (B') Logística, τ val. | 0,708 ± 0,006 | 0,937 ± 0,005 | 43,1 ± 1,0 | −0,099 [−0,121; −0,078] | 0,785 ± 0,006 | −0,075 [−0,104; −0,046] |

**Lectura honesta.** Fuera de su generador, **la ventaja de los modelos probabilísticos desaparece**: las reglas
de Albert quedan levemente por encima del HMM A' y claramente por encima de la logística B'. El HMM se degrada
menos que la logística (en G3-guion+agentes empata con las reglas: ΔF1 = +0,006 [−0,018; +0,031]), lo que sugiere
que el estado oculto generaliza mejor que una ventana de conteos. La causa principal es el **sensor**: en G3 el
32–47 % de los turnos no activa ningún atajo (16 % en G1) y solo el 1–3 % usa el léxico negativo de Albert
(10 % en G1). Los usuarios se frustran con palabras que el sensor no conoce, y los modelos, que aprendieron
a esperar «urgente» o «pésimo», sobre-escalan con mensajes «sin atajo» acumulados (error dominante de B').
Conclusión: lo que se debe mejorar primero es el sensor, no el decisor. Errores por tipo en
`experiments/externa_errores_por_tipo.csv` y 36 casos con texto en `experiments/externa_errores_muestra.csv`.

**Prueba real (piloto de Albert, n = 487; etiqueta proxy «terminó con asesor»).** Export del panel del 6 oct. al
27 nov. de 2026: 500 conversaciones; se excluyeron 13 con un mensaje de usuario vacío. 49 de 487 (10,1 %) terminaron
con asesor. Modelos entrenados con G1 (semilla 42) y umbrales de la validación sintética; el piloto solo se evalúa.
Al repositorio solo llegan métricas agregadas (`experiments/piloto_resumen.md`, `piloto_resultados.json`,
`piloto_errores_por_tipo.csv`).

| Modelo | Recall [IC95] | Precisión [IC95] | F1 | % innec. | VP / FP / FN | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|
| Reglas fijas (Albert) | 0,837 [0,71; 0,91] | 0,165 [0,12; 0,22] | 0,275 | 83,5 | 41 / 208 / 8 | — |
| Clase mayoritaria | 0,000 | — | 0,000 | — | 0 / 0 / 49 | — |
| (A) HMM, τ* = 0,20 | 0,939 [0,83; 0,98] | 0,123 [0,09; 0,16] | 0,217 | 87,7 | 46 / 328 / 3 | −0,058 [−0,098; −0,018] |
| (A') HMM, τ = 0,75 (val.) | 0,531 [0,39; 0,66] | 0,138 [0,10; 0,19] | 0,219 | 86,2 | 26 / 162 / 23 | −0,056 [−0,109; −0,006] |
| (B) Logística, τ* = 0,20 | 0,939 [0,83; 0,98] | 0,124 [0,09; 0,16] | 0,219 | 87,6 | 46 / 325 / 3 | −0,057 [−0,097; −0,016] |
| (B') Logística, τ = 0,40 (val.) | 0,857 [0,73; 0,93] | 0,138 [0,10; 0,18] | 0,238 | 86,2 | 42 / 262 / 7 | −0,038 [−0,084; +0,010] |

**Lectura honesta.**

* **Ningún modelo supera a las reglas** en datos reales: el HMM A' y las dos variantes predeclaradas quedan por debajo
  con un IC que excluye el cero, y la logística B' empata (su IC incluye el cero). Ninguno cumple F1 ≥ 0,80,
  innecesarios ≤ 20 % ni ECE ≤ 0,10 (ECE 0,26–0,44).
* **El proxy no sirve para medir precisión.** Las reglas simuladas habrían escalado 249 conversaciones, pero solo 49
  terminaron con asesor: un asesor puede no haber atendido aunque hiciera falta. Por eso todas las precisiones quedan
  por debajo de 0,17. La etiqueta manual ciega (`--plantilla-etiquetas`) queda pendiente y es la que haría concluyente
  esta prueba.
* **Se repite el diagnóstico de G3: el sensor.** El 50,3 % de los turnos reales no activa ningún atajo (16 % en G1) y
  solo el 1,3 % usa léxico negativo (10 % en G1); el error dominante de los modelos es el falso positivo por mensajes
  «sin atajo» acumulados.
* **Desvíos frente al Hito 1:** n = 487 en lugar de 120, 53 días en lugar de 31, 10,1 % con asesor en lugar de 36 %,
  exclusión de 13 conversaciones con mensaje vacío y etiqueta proxy en lugar de manual.
* La anonimización hizo 17 reemplazos en 1 734 turnos de usuario; el autor revisó `revision_manual.txt` completo a mano
  (paso 4 de la guía).

## Licencia

Código: MIT (`LICENSE`). Datos sintéticos: CC BY 4.0. Datos del piloto: propiedad de la UPC, no incluidos.
