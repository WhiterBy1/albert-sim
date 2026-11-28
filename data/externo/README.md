# data/externo · Prueba externa G3 (2 000 conversaciones sintéticas de «IPS X»)

Conversaciones **100 % sintéticas** entre usuarios y el chatbot de una IPS ficticia («IPS X»: laboratorio
clínico y patología). Se generaron **aparte** del generador del proyecto (`src/generar_datos.py`), con otros
supuestos, para medir si los modelos entrenados con G1 se sostienen fuera de su propio generador.
Se usan **solo para evaluar** (`python -m src.prueba_externa`); nunca para entrenar ni elegir umbrales.

Licencia: CC BY 4.0. No hay datos personales: los documentos usan un rango ficticio que empieza por `99`,
las órdenes son aleatorias (`ORD-XXXXXX`) y las direcciones y teléfonos son inventados.

## Variantes

| Variante | n | Cómo se generó | % requiere asesor |
|---|---|---|---|
| `G3-guion` | 500 | Generadas con 12 escenarios guionados, perfiles de escritura y ruido (tipeo, mayúsculas) | 40,2 |
| `G3-agentes` | 500 | Generadas con un usuario simulado con objetivos y **paciencia** frente a un bot de reglas por palabras clave; la frustración emerge de la interacción | 32,6 |
| `G3-llm` | 1 000 | Generadas con **dos agentes LLM** (Claude Haiku 5.5 vía API de Anthropic): bot con base de conocimiento fija y respuestas libres; usuario con persona, objetivos ocultos y paciencia que controla el simulador | 50,6 |

## Etiqueta

La calcula el **simulador**, no el modelo de lenguaje. `requiere_asesor = 1` si el simulador marcó un momento
en que el bot ya no podía resolver (objetivo que solo atiende un humano, como interpretar resultados, trámites
de EPS o adelantar entregas, o dos fallos seguidos en un objetivo importante) o si el usuario pidió un humano.
`turno_oraculo` es el primero de esos momentos, contado en turnos de usuario.
`termino_con_asesor` (proxy) indica si el bot transfirió.

## Preprocesamiento (de los archivos originales a estos CSV)

* Solo los mensajes del **usuario** (lo que ve el sensor de Albert); se excluye la respuesta a la encuesta
  de satisfacción, que Albert no tiene.
* Las notas de voz se reemplazan por `[NOTA_DE_VOZ]` sin texto, igual que en G1/G2 y en Albert.

## Archivos

* `conversaciones.csv`: `conversacion_id`, `variante`, `generador`, `perfil` (adulto, joven, apurado, adulto_mayor),
  `canal`, `requiere_asesor`, `turno_oraculo`, `n_turnos`, `termino_con_asesor`, `etiqueta_satisfaccion`, `usa_voz`.
* `turnos.csv`: `conversacion_id`, `turno`, `texto`, `es_voz`.

La variante LLM **no es determinista**; por eso los CSV se versionan y son la referencia para reproducir la prueba G3.

## Costo de la API (variante G3-llm)

1 000 conversaciones ≈ 14 000 llamadas a Claude Haiku 5.5 (dos agentes por turno), ≈ 19 M tokens de entrada
y ≈ 2 M de salida (estimado a partir del número de turnos; el uso exacto se lee en la consola de Anthropic).
Las variantes `G3-guion` y `G3-agentes` no usan APIs.
