# Resultados (media ± desv. estándar, 5 semillas)

Semillas: 42, 43, 44, 45, 46. Umbrales de éxito (Hito 1): {"recall": 0.9, "f1": 0.8, "pct_innecesarios": 20.0, "retraso_mediana": 1.0, "ece": 0.1, "brecha_f1_subgrupo": 0.1}

τ* = regla predeclarada (0,20; solo baja si en validación recall < 0,90). «τ en validación» = desvío justificado (máx. F1 con recall ≥ 0,90 en validación). Los hiperparámetros (α del HMM, C de la logística) se eligen solo en validación.

## Prueba independiente G2 (métrica principal)

| Modelo | recall | f1 | precision | pct_innecesarios | retraso_mediana | ece | brecha_f1_max |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.890 ± 0.030 | 0.819 ± 0.026 | 0.758 ± 0.026 | 24.153 ± 2.593 | 1.000 ± 0.000 | — | 0.079 ± 0.031 |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — | 0.016 ± 0.018 | 0.000 ± 0.000 |
| (A) HMM · τ* predeclarado | 1.000 ± 0.000 | 0.708 ± 0.013 | 0.548 ± 0.015 | 45.183 ± 1.522 | 0.000 ± 0.000 | 0.120 ± 0.007 | 0.119 ± 0.043 |
| (A') HMM · τ en validación | 0.939 ± 0.058 | 0.914 ± 0.027 | 0.895 ± 0.065 | 10.511 ± 6.476 | 0.800 ± 0.447 | 0.132 ± 0.025 | 0.133 ± 0.050 |
| (B) Logística · τ* predeclarado | 0.998 ± 0.005 | 0.797 ± 0.032 | 0.665 ± 0.043 | 33.541 ± 4.283 | 0.000 ± 0.000 | 0.149 ± 0.005 | 0.145 ± 0.028 |
| (B') Logística · τ en validación | 0.982 ± 0.013 | 0.919 ± 0.007 | 0.863 ± 0.009 | 13.737 ± 0.871 | 0.000 ± 0.000 | 0.112 ± 0.009 | 0.160 ± 0.023 |

## Prueba G1 (misma familia que el entrenamiento)

| Modelo | recall | f1 | precision | pct_innecesarios | retraso_mediana | ece | brecha_f1_max |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.886 ± 0.026 | 0.798 ± 0.037 | 0.728 ± 0.059 | 27.180 ± 5.863 | 0.800 ± 0.447 | — | 0.132 ± 0.044 |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — | 0.021 ± 0.018 | 0.000 ± 0.000 |
| (A) HMM · τ* predeclarado | 0.998 ± 0.004 | 0.714 ± 0.034 | 0.556 ± 0.042 | 44.366 ± 4.191 | 0.000 ± 0.000 | 0.102 ± 0.010 | 0.144 ± 0.045 |
| (A') HMM · τ en validación | 0.936 ± 0.040 | 0.930 ± 0.021 | 0.929 ± 0.070 | 7.056 ± 7.027 | 0.900 ± 0.224 | 0.144 ± 0.013 | 0.102 ± 0.022 |
| (B) Logística · τ* predeclarado | 1.000 ± 0.000 | 0.845 ± 0.052 | 0.735 ± 0.081 | 26.489 ± 8.073 | 0.000 ± 0.000 | 0.154 ± 0.014 | 0.084 ± 0.016 |
| (B') Logística · τ en validación | 0.996 ± 0.009 | 0.967 ± 0.010 | 0.940 ± 0.018 | 5.957 ± 1.750 | 0.000 ± 0.000 | 0.131 ± 0.016 | 0.101 ± 0.041 |

## Validación (solo para elegir hiperparámetros; optimista)

| Modelo | recall | f1 | precision | pct_innecesarios | retraso_mediana | ece | brecha_f1_max |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.861 ± 0.024 | 0.797 ± 0.025 | 0.744 ± 0.046 | 25.585 ± 4.609 | 1.000 ± 0.000 | — | 0.099 ± 0.048 |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — | 0.034 ± 0.025 | 0.000 ± 0.000 |
| (A) HMM · τ* predeclarado | 1.000 ± 0.000 | 0.699 ± 0.060 | 0.540 ± 0.069 | 46.020 ± 6.898 | 0.000 ± 0.000 | 0.091 ± 0.023 | 0.137 ± 0.049 |
| (A') HMM · τ en validación | 0.930 ± 0.023 | 0.926 ± 0.037 | 0.927 ± 0.080 | 7.327 ± 7.985 | 1.000 ± 0.000 | 0.143 ± 0.008 | 0.107 ± 0.037 |
| (B) Logística · τ* predeclarado | 1.000 ± 0.000 | 0.818 ± 0.067 | 0.696 ± 0.097 | 30.367 ± 9.720 | 0.000 ± 0.000 | 0.143 ± 0.030 | 0.140 ± 0.026 |
| (B') Logística · τ en validación | 0.992 ± 0.008 | 0.967 ± 0.008 | 0.942 ± 0.022 | 5.762 ± 2.226 | 0.000 ± 0.000 | 0.133 ± 0.012 | 0.098 ± 0.032 |

## Validación cruzada de 5 particiones por conversación (G1, semilla 42; media ± desv. entre particiones)

| Modelo | recall | f1 | precision | pct_innecesarios | retraso_mediana | ece | brecha_f1_max |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.875 ± 0.029 | 0.815 ± 0.028 | 0.764 ± 0.041 | 23.647 ± 4.111 | 1.000 ± 0.000 | — | 0.099 ± 0.028 |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | — | 0.017 ± 0.017 | 0.000 ± 0.000 |
| (A) HMM · τ* predeclarado | 1.000 ± 0.000 | 0.738 ± 0.028 | 0.586 ± 0.035 | 41.416 ± 3.518 | 0.000 ± 0.000 | 0.088 ± 0.011 | 0.116 ± 0.046 |
| (A') HMM · τ en validación | 0.928 ± 0.011 | 0.949 ± 0.006 | 0.972 ± 0.016 | 2.827 ± 1.634 | 1.000 ± 0.000 | 0.142 ± 0.008 | 0.089 ± 0.028 |
| (B) Logística · τ* predeclarado | 1.000 ± 0.000 | 0.827 ± 0.041 | 0.707 ± 0.061 | 29.346 ± 6.127 | 0.000 ± 0.000 | 0.146 ± 0.008 | 0.115 ± 0.042 |
| (B') Logística · τ en validación | 0.988 ± 0.008 | 0.966 ± 0.008 | 0.946 ± 0.014 | 5.423 ± 1.432 | 0.000 ± 0.000 | 0.121 ± 0.018 | 0.080 ± 0.026 |
