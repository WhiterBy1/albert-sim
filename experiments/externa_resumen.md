# Prueba externa G3 (generador independiente; los modelos se entrenan solo con G1)

Media ± desv. en 5 semillas de entrenamiento (42–46). ΔF1 vs reglas: bootstrap pareado (2 000 remuestreos, semilla 42).

## G3-llm · etiqueta: simulador (n = 1000; 50.6 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.818 ± 0.000 | 0.857 ± 0.000 | 0.900 ± 0.000 | 10.0 ± 0.0 | 2.0 ± 0.0 | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.161 ± 0.009 | — |
| (A) HMM · τ* predeclarado | 0.937 ± 0.000 | 0.797 ± 0.001 | 0.694 ± 0.002 | 30.6 ± 0.2 | 0.0 ± 0.0 | 0.191 ± 0.007 | -0.058 [-0.085, -0.032] |
| (A') HMM · τ en validación | 0.840 ± 0.039 | 0.845 ± 0.011 | 0.853 ± 0.022 | 14.7 ± 2.2 | 1.4 ± 0.5 | 0.106 ± 0.021 | -0.031 [-0.052, -0.011] |
| (B) Logística · τ* predeclarado | 0.963 ± 0.005 | 0.779 ± 0.003 | 0.655 ± 0.004 | 34.5 ± 0.4 | 0.0 ± 0.0 | 0.144 ± 0.019 | -0.081 [-0.110, -0.053] |
| (B') Logística · τ en validación | 0.953 ± 0.006 | 0.785 ± 0.006 | 0.668 ± 0.012 | 33.2 ± 1.2 | 0.0 ± 0.0 | 0.079 ± 0.016 | -0.075 [-0.104, -0.046] |

## G3-guion+agentes · etiqueta: simulador (n = 1000; 36.4 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.745 ± 0.000 | 0.736 ± 0.000 | 0.728 ± 0.000 | 27.2 ± 0.0 | 1.0 ± 0.0 | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.019 ± 0.009 | — |
| (A) HMM · τ* predeclarado | 0.918 ± 0.000 | 0.601 ± 0.001 | 0.447 ± 0.002 | 55.3 ± 0.2 | 0.0 ± 0.0 | 0.054 ± 0.008 | -0.134 [-0.167, -0.099] |
| (A') HMM · τ en validación | 0.836 ± 0.037 | 0.732 ± 0.010 | 0.653 ± 0.038 | 34.7 ± 3.8 | 0.4 ± 0.5 | 0.171 ± 0.019 | +0.006 [-0.018, +0.031] |
| (B) Logística · τ* predeclarado | 0.931 ± 0.009 | 0.590 ± 0.010 | 0.432 ± 0.013 | 56.8 ± 1.3 | 0.0 ± 0.0 | 0.231 ± 0.012 | -0.151 [-0.187, -0.115] |
| (B') Logística · τ en validación | 0.915 ± 0.005 | 0.620 ± 0.006 | 0.469 ± 0.009 | 53.1 ± 0.9 | 0.0 ± 0.0 | 0.230 ± 0.009 | -0.118 [-0.153, -0.085] |

## G3 total · etiqueta: simulador (n = 2000; 43.5 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.787 ± 0.000 | 0.805 ± 0.000 | 0.823 ± 0.000 | 17.7 ± 0.0 | 1.0 ± 0.0 | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.090 ± 0.009 | — |
| (A) HMM · τ* predeclarado | 0.929 ± 0.000 | 0.702 ± 0.001 | 0.565 ± 0.002 | 43.5 ± 0.2 | 0.0 ± 0.0 | 0.122 ± 0.008 | -0.101 [-0.122, -0.080] |
| (A') HMM · τ en validación | 0.838 ± 0.038 | 0.794 ± 0.005 | 0.756 ± 0.030 | 24.4 ± 3.0 | 1.0 ± 0.0 | 0.113 ± 0.010 | -0.017 [-0.033, -0.001] |
| (B) Logística · τ* predeclarado | 0.949 ± 0.006 | 0.688 ± 0.007 | 0.540 ± 0.010 | 46.0 ± 1.0 | 0.0 ± 0.0 | 0.173 ± 0.011 | -0.120 [-0.143, -0.098] |
| (B') Logística · τ en validación | 0.937 ± 0.005 | 0.708 ± 0.006 | 0.569 ± 0.010 | 43.1 ± 1.0 | 0.0 ± 0.0 | 0.144 ± 0.023 | -0.099 [-0.121, -0.078] |

## G3-llm · etiqueta: proxy_termino_con_asesor (n = 1000; 33.5 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.606 ± 0.000 | 0.511 ± 0.000 | 0.441 ± 0.000 | 55.9 ± 0.0 | — | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.011 ± 0.008 | — |
| (A) HMM · τ* predeclarado | 0.879 ± 0.002 | 0.579 ± 0.000 | 0.431 ± 0.000 | 56.9 ± 0.0 | — | 0.066 ± 0.002 | — |
| (A') HMM · τ en validación | 0.672 ± 0.061 | 0.539 ± 0.026 | 0.451 ± 0.009 | 54.9 ± 0.9 | — | 0.199 ± 0.039 | — |
| (B) Logística · τ* predeclarado | 0.935 ± 0.012 | 0.580 ± 0.004 | 0.421 ± 0.002 | 57.9 ± 0.2 | — | 0.179 ± 0.027 | — |
| (B') Logística · τ en validación | 0.908 ± 0.012 | 0.576 ± 0.001 | 0.422 ± 0.004 | 57.8 ± 0.4 | — | 0.226 ± 0.034 | — |

## G3-guion+agentes · etiqueta: proxy_termino_con_asesor (n = 1000; 8.0 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.988 ± 0.000 | 0.350 ± 0.000 | 0.212 ± 0.000 | 78.8 ± 0.0 | 1.0 ± 0.0 | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.265 ± 0.009 | — |
| (A) HMM · τ* predeclarado | 1.000 ± 0.000 | 0.193 ± 0.001 | 0.107 ± 0.000 | 89.3 ± 0.0 | 0.0 ± 0.0 | 0.233 ± 0.009 | — |
| (A') HMM · τ en validación | 0.993 ± 0.007 | 0.291 ± 0.024 | 0.171 ± 0.016 | 82.9 ± 1.6 | 1.0 ± 0.0 | 0.412 ± 0.027 | — |
| (B) Logística · τ* predeclarado | 1.000 ± 0.000 | 0.185 ± 0.006 | 0.102 ± 0.004 | 89.8 ± 0.4 | 0.0 ± 0.0 | 0.362 ± 0.055 | — |
| (B') Logística · τ en validación | 1.000 ± 0.000 | 0.202 ± 0.004 | 0.113 ± 0.003 | 88.7 ± 0.3 | 0.0 ± 0.0 | 0.495 ± 0.009 | — |

## G3 total · etiqueta: proxy_termino_con_asesor (n = 2000; 20.8 % requiere asesor)

| Modelo | Recall | F1 | Precisión | % innecesarios | Retraso (mediana) | ECE | ΔF1 vs reglas [IC95] |
|---|---|---|---|---|---|---|---|
| Línea base 1 · Reglas fijas (Albert) | 0.680 ± 0.000 | 0.452 ± 0.000 | 0.339 ± 0.000 | 66.1 ± 0.0 | — | — | — |
| Línea base 2 · Clase mayoritaria | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0 ± 0.0 | — | 0.138 ± 0.009 | — |
| (A) HMM · τ* predeclarado | 0.903 ± 0.001 | 0.406 ± 0.000 | 0.262 ± 0.000 | 73.8 ± 0.0 | — | 0.129 ± 0.010 | — |
| (A') HMM · τ en validación | 0.734 ± 0.051 | 0.441 ± 0.008 | 0.315 ± 0.007 | 68.5 ± 0.7 | — | 0.292 ± 0.033 | — |
| (B) Logística · τ* predeclarado | 0.947 ± 0.010 | 0.404 ± 0.005 | 0.257 ± 0.004 | 74.3 ± 0.4 | — | 0.233 ± 0.055 | — |
| (B') Logística · τ en validación | 0.926 ± 0.010 | 0.416 ± 0.003 | 0.268 ± 0.003 | 73.2 ± 0.3 | — | 0.353 ± 0.022 | — |

