-- Esquema reducido del panel de Albert (solo lo necesario para el proyecto).
-- Ningún campo guarda datos personales: el contacto es un hash y el texto está anonimizado.
CREATE TABLE IF NOT EXISTS conversacion (
    conversacion_id   TEXT PRIMARY KEY,
    contacto_hash     CHAR(64)  NOT NULL,           -- SHA-256 con sal (nunca el teléfono)
    fuente            TEXT      NOT NULL,           -- 'sintetico' | 'piloto_anonimizado'
    seed              INTEGER,
    variante          TEXT,                         -- G1 | G2
    particion         TEXT,                         -- train | val | test_g1 | test_g2 | piloto
    intencion         TEXT,
    tipo_usuario      TEXT,                         -- titular | tercero
    escritura         TEXT,                         -- buena | mala
    usa_voz           SMALLINT,
    caso_complejo     SMALLINT,
    requiere_asesor   SMALLINT  NOT NULL,           -- etiqueta
    turno_oraculo     INTEGER,                      -- turno en que la necesidad se hizo evidente
    n_turnos          INTEGER
);

CREATE TABLE IF NOT EXISTS mensaje (
    mensaje_id        BIGSERIAL PRIMARY KEY,
    conversacion_id   TEXT      NOT NULL REFERENCES conversacion(conversacion_id) ON DELETE CASCADE,
    turno             INTEGER   NOT NULL,
    texto             TEXT      NOT NULL,           -- ya anonimizado
    es_voz            SMALLINT  NOT NULL DEFAULT 0,
    obs               TEXT      NOT NULL,           -- símbolo discreto del sensor
    s_pide_asesor     SMALLINT, s_negativo SMALLINT, s_repite SMALLINT,
    s_voz             SMALLINT, s_sin_atajo SMALLINT, s_cierre SMALLINT, s_atajo SMALLINT,
    estado_latente    SMALLINT,                     -- solo existe en datos sintéticos
    frustracion       REAL,
    UNIQUE (conversacion_id, turno)
);

CREATE INDEX IF NOT EXISTS ix_conv_particion ON conversacion (particion);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON mensaje (conversacion_id, turno);
