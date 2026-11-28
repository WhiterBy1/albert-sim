# Prueba real · piloto de Albert (n = 487, etiqueta: proxy_termino_con_asesor)

| Modelo | Recall [IC95] | Precisión [IC95] | F1 | % innecesarios | Retraso (mediana) | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.837 [0.71, 0.91] | 0.165 [0.12, 0.22] | 0.275 | 83.5 | 0.0 | — |
| Línea base 2 · Clase mayoritaria | 0.000 [0.00, 0.07] | 0.000 [0.00, 0.00] | 0.000 | 0.0 | nan | — |
| (A) HMM · τ* predeclarado | 0.939 [0.83, 0.98] | 0.123 [0.09, 0.16] | 0.217 | 87.7 | 0.0 | -0.058 [-0.098, -0.018] |
| (A') HMM · τ en validación | 0.531 [0.39, 0.66] | 0.138 [0.10, 0.19] | 0.219 | 86.2 | 0.0 | -0.056 [-0.109, -0.006] |
| (B) Logística · τ* predeclarado | 0.939 [0.83, 0.98] | 0.124 [0.09, 0.16] | 0.219 | 87.6 | 0.0 | -0.057 [-0.097, -0.016] |
| (B') Logística · τ en validación | 0.857 [0.73, 0.93] | 0.138 [0.10, 0.18] | 0.238 | 86.2 | 0.0 | -0.038 [-0.084, +0.010] |
