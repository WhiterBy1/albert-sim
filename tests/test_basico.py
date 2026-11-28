"""Pruebas mínimas: reproducibilidad, partición sin fuga y anonimización."""
import pandas as pd

from src.anonimizar_piloto import anonimizar, anonimizar_texto
from src.generar_datos import construir
from src.modelos import ReglasFijas
from src.observaciones import observar


def test_generador_reproducible():
    a, _ = construir(7, n_g1=50, n_g2=10, guardar=False)
    b, _ = construir(7, n_g1=50, n_g2=10, guardar=False)
    pd.testing.assert_frame_equal(a, b)


def test_particion_por_conversacion():
    c, t = construir(7, n_g1=200, n_g2=20, guardar=False)
    assert c.conversacion_id.is_unique
    assert t.merge(c[["conversacion_id", "particion"]]).groupby("conversacion_id").particion.nunique().max() == 1


def test_sensor():
    assert observar("Quiero hablar con un asesor", False, None)["obs"] == "PIDE_ASESOR"
    assert observar("¿A qué hora abren?", False, None)["obs"] == "ATAJO"
    assert observar("[NOTA_DE_VOZ]", True, None)["obs"] == "VOZ"


def test_reglas_tres_fallos():
    dec, _ = ReglasFijas().decidir(["ATAJO", "SIN_ATAJO", "VOZ", "SIN_ATAJO"])
    assert dec == [0, 0, 0, 1]


def test_anonimizacion_no_deja_datos():
    t, n = anonimizar_texto("Soy Laura Gómez, cédula 1.047.123.456, cel 300 123 4567, laura@x.com, orden No. A-55821")
    for dato in ("Laura", "Gómez", "1.047", "300 123", "laura@", "55821"):
        assert dato not in t
    assert n >= 5


def test_anonimizacion_export_ficticio():
    df = pd.read_csv("tests/ejemplo_export_ficticio.csv")
    convs, turnos = anonimizar(df, sal="sal-de-prueba-1234567890")
    texto = " ".join(turnos.texto)
    for dato in ("Laura", "Rosa", "1.047.123.456", "45123987", "6056789", "Calle 30", "laura.g@", "02/09/2026"):
        assert dato not in texto, dato
    assert convs.requiere_asesor.tolist() == [1, 0, 0]
    assert convs.turno_oraculo.iloc[0] == 3
    assert not convs.contacto_hash.str.contains("300").any()


def test_filtrado_coincide_con_hmmlearn():
    """El último paso del filtrado debe coincidir con el posterior de hmmlearn."""
    import numpy as np
    from hmmlearn.hmm import CategoricalHMM

    from src.generar_datos import construir
    from src.modelos import HMMEscalamiento
    from src.observaciones import IDX

    c, t = construir(3, n_g1=300, n_g2=10, guardar=False)
    m = HMMEscalamiento().ajustar(c, t)
    ref = CategoricalHMM(n_components=3, n_features=len(IDX), init_params="", params="")
    ref.startprob_, ref.transmat_, ref.emissionprob_ = m.pi, m.A, m.B
    for _, g in list(t.groupby("conversacion_id"))[:30]:
        obs = g.obs.tolist()
        esperado = ref.predict_proba(np.array([[IDX[o]] for o in obs]))[-1, 2]
        assert abs(m.filtrar(obs)[-1] - esperado) < 1e-8


def test_logistica_no_mira_el_futuro():
    """Los rasgos del turno t no cambian si se agregan turnos posteriores (decisión en línea)."""
    import numpy as np

    from src.modelos import RegresionLogistica

    obs = ["ATAJO", "SIN_ATAJO", "REPITE", "VOZ", "NEGATIVO", "PIDE_ASESOR", "SIN_ATAJO"]
    m = RegresionLogistica()
    completo = m.rasgos(obs, tercero=1)
    for t in range(1, len(obs) + 1):
        assert np.allclose(m.rasgos(obs[:t], tercero=1)[-1], completo[t - 1])


def test_modelos_y_umbral_en_validacion():
    """La logística y el HMM devuelven probabilidades en [0, 1] y el umbral elegido sale de la rejilla."""
    from src.generar_datos import construir
    from src.modelos import TAUS, HMMEscalamiento, RegresionLogistica, tau_en_validacion

    c, t = construir(5, n_g1=300, n_g2=10, guardar=False)
    tr, va = c[c.particion == "train"], c[c.particion == "val"]
    ttr, tva = t[t.conversacion_id.isin(tr.conversacion_id)], t[t.conversacion_id.isin(va.conversacion_id)]
    for m in (HMMEscalamiento().ajustar(tr, ttr), RegresionLogistica(c=1.0).ajustar(tr, ttr)):
        _, p = m.decidir(tva.obs.tolist()[:6], tercero=0)
        assert all(0.0 <= x <= 1.0 for x in p)
        assert tau_en_validacion(m, va, tva) in TAUS


def test_prueba_externa_formato():
    """G3: etiquetas binarias, turno oráculo dentro de la conversación y voz sin texto."""
    from src.prueba_externa import cargar_externa
    c, t = cargar_externa()
    assert len(c) == 2000 and set(c.requiere_asesor.unique()) <= {0, 1}
    pos = c[c.requiere_asesor == 1]
    assert pos.turno_oraculo.notna().all() and (pos.turno_oraculo <= pos.n_turnos).all()
    assert (t[t.es_voz == 1].texto == "[NOTA_DE_VOZ]").all()
    assert set(t.obs) <= {"PIDE_ASESOR", "NEGATIVO", "REPITE", "VOZ", "SIN_ATAJO", "CIERRE", "ATAJO"}
