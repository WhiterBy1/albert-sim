# data/

Aquí **no hay datos personales**. Hay dos tipos de datos:

| Carpeta | Qué contiene | ¿Se sube al repositorio? | Licencia |
|---|---|---|---|
| `seed_<n>/` | Conversaciones sintéticas (`conversaciones.csv`, `turnos.csv`) | No: se regeneran con `python -m src.generar_datos --seed <n>` (deterministas) | CC BY 4.0 |
| `piloto/` | Piloto de Albert anonimizado (120 conversaciones, 31 días) | **Nunca**: queda dentro de la UPC | Uso autorizado por la UPC |

## Diccionario de datos

**conversaciones.csv** — una fila por conversación: `conversacion_id`, `variante` (G1 entrenamiento/validación, G2 prueba independiente), `particion` (train, val, test_g1, test_g2), `intencion`, `tipo_usuario` (titular/tercero), `escritura` (buena/mala), `usa_voz`, `caso_complejo`, `requiere_asesor` (**etiqueta**), `turno_oraculo` (turno en que la necesidad se hizo evidente), `n_turnos`.

**turnos.csv** — una fila por mensaje del usuario: `conversacion_id`, `turno`, `texto`, `es_voz`, señales del sensor (`s_pide_asesor`, `s_negativo`, `s_repite`, `s_voz`, `s_sin_atajo`, `s_cierre`, `s_atajo`), `obs` (símbolo discreto), `estado_latente` (1 encaminado, 2 atascado, 3 requiere asesor; solo en sintéticos) y `frustracion` (variable interna del generador).

## Calibración con el piloto

El generador solo toma **conteos agregados** del piloto: 36 % de conversaciones terminó con asesor y 23 % de los mensajes cayó al LLM (sin atajo). Ningún texto ni etiqueta del piloto se usa para generar ni para entrenar.
