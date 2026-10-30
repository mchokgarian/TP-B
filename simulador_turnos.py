# ============================================================
# TP2b PUERTO SUR - GRUPO 11
# CODIGO 6: SIMULADOR CON TURNOS - VERSION CORREGIDA
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import simpy

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "PuertoSur_escalas_G11.csv"
SALIDA = BASE / "Resultados_Simulador_Turnos"

SALIDA.mkdir(exist_ok=True)

# PARAMETROS

BERMAS = 3
STS = 3

TASA = {
    "mov_dry": 28.0,
    "mov_reefer": 21.0,
    "mov_oog": 12.0
}

SEMILLA = 51
RNG = np.random.default_rng(SEMILLA)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

def cargar_datos():

    df = pd.read_csv(
        ARCHIVO,
        parse_dates=["fecha_hora_arribo"]
    )

    df = df.sort_values(
        "fecha_hora_arribo"
    ).reset_index(drop=True)

    columnas = [
        "id_escala",
        "fecha_hora_arribo",
        "servicio",
        "mov_dry",
        "mov_reefer",
        "mov_oog"
    ]

    if df[columnas].isna().any().any():
        raise ValueError("Existen datos faltantes")

    if df["id_escala"].duplicated().any():
        raise ValueError("Existen identificadores duplicados")

    origen = df["fecha_hora_arribo"].iloc[0]

    df["hora_sim"] = (
        df["fecha_hora_arribo"] - origen
    ).dt.total_seconds() / 3600

    return df, origen


# ============================================================
# 2. CLASE PUERTO
# ============================================================

class Puerto:

    def __init__(self, env, origen):

        self.env = env

        self.origen_h = (
            origen.hour + origen.minute / 60
        )

        self.bermas = simpy.Resource(
            env,
            capacity=BERMAS
        )

        self.gruas = simpy.Resource(
            env,
            capacity=STS
        )

        self.nocturno = simpy.Resource(
            env,
            capacity=2
        )

        self.registros = []

        self.uso_bermas_h = 0.0
        self.uso_gruas_h = 0.0

        self.duraciones_relevo = {}
        self.standby_listo = {}

        self.activaciones = 0


    # --------------------------------------------------------
    # TIEMPO
    # --------------------------------------------------------

    def tiempo_calendario(self):

        return self.origen_h + self.env.now


    def tramo_actual(self):

        t = self.tiempo_calendario()

        dia = int(t // 24)
        hora = t % 24

        inicio = dia * 24

        if hora < 8:

            return inicio, inicio + 8, True

        elif hora < 16:

            return inicio + 8, inicio + 16, False

        else:

            return inicio + 16, inicio + 24, False


    # --------------------------------------------------------
    # CAMBIOS DE TURNO
    # --------------------------------------------------------

    def relevo(self, inicio_turno):

        if inicio_turno not in self.duraciones_relevo:

            self.duraciones_relevo[inicio_turno] = (
                RNG.uniform(10, 15) / 60
            )

        return self.duraciones_relevo[inicio_turno]


    def esperar_relevo(self):

        inicio, fin, noche = self.tramo_actual()

        demora = (
            inicio
            + self.relevo(inicio)
            - self.tiempo_calendario()
        )

        if demora > 1e-9:

            yield self.env.timeout(demora)


    # ========================================================
    # 3. OPERACION DE GRUAS
    # ========================================================

    def operar(self, fila):

        pendientes = sum(
            float(fila[k]) / tasa
            for k, tasa in TASA.items()
        )

        activo = 0.0

        while pendientes > 1e-8:

            yield from self.esperar_relevo()

            inicio, fin, noche = self.tramo_actual()

            tiempo_disponible = (
                fin - self.tiempo_calendario()
            )

            if tiempo_disponible <= 1e-8:

                yield self.env.timeout(1e-6)
                continue

            # -----------------------------------------------
            # TURNO NOCTURNO
            # -----------------------------------------------

            if noche:

                with self.nocturno.request() as permiso_noche:

                    yield permiso_noche

                    clave = inicio

                    # Activar standby cuando se requiere
                    # una segunda grua simultanea.

                    if self.nocturno.count == 2:

                        if clave not in self.standby_listo:

                            self.standby_listo[clave] = (
                                self.env.now
                                + RNG.uniform(15, 20) / 60
                            )

                            self.activaciones += 1

                        restante = (
                            self.standby_listo[clave]
                            - self.env.now
                        )

                        if restante > 0:

                            yield self.env.timeout(restante)

                    if self.tiempo_calendario() >= fin - 1e-8:
                        continue

                    with self.gruas.request() as permiso_grua:

                        yield permiso_grua

                        duracion = min(
                            pendientes,
                            fin - self.tiempo_calendario()
                        )

                        if duracion > 1e-8:

                            yield self.env.timeout(duracion)

                            pendientes -= duracion
                            activo += duracion

                            self.uso_gruas_h += duracion

            # -----------------------------------------------
            # TURNOS DIURNO Y VESPERTINO
            # -----------------------------------------------

            else:

                with self.gruas.request() as permiso_grua:

                    yield permiso_grua

                    duracion = min(
                        pendientes,
                        fin - self.tiempo_calendario()
                    )

                    if duracion > 1e-8:

                        yield self.env.timeout(duracion)

                        pendientes -= duracion
                        activo += duracion

                        self.uso_gruas_h += duracion

        return activo


    # ========================================================
    # 4. ATENCION DE BUQUES
    # ========================================================

    def atender(self, fila):

        llegada = self.env.now

        with self.bermas.request() as permiso:

            yield permiso

            espera = self.env.now - llegada

            entrada = self.env.now

            maniobras = RNG.uniform(1, 2)

            # Atraque

            yield self.env.timeout(maniobras / 2)

            # Operacion

            inicio_op = self.env.now

            operacion_activa = yield from self.operar(fila)

            fin_op = self.env.now

            # Desatraque

            yield self.env.timeout(maniobras / 2)

            salida = self.env.now

            self.uso_bermas_h += salida - entrada

        self.registros.append({

            "id_escala": fila["id_escala"],

            "servicio": fila["servicio"],

            "fecha_hora_arribo": fila["fecha_hora_arribo"],

            "espera_rada_h": espera,

            "maniobras_h": maniobras,

            "operacion_efectiva_h": operacion_activa,

            "operacion_transcurrida_h": (
                fin_op - inicio_op
            ),

            "estadia_total_h": salida - llegada,

            "hora_salida_sim": salida

        })


# ============================================================
# 5. GENERADOR DE LLEGADAS
# ============================================================

def llegadas(env, puerto, df):

    for i, fila in df.iterrows():

        yield env.timeout(
            max(
                0,
                float(fila["hora_sim"]) - env.now
            )
        )

        env.process(
            puerto.atender(fila)
        )

        if (i + 1) % 1000 == 0:

            print(
                f"Ingresaron {i + 1} de {len(df)} escalas",
                flush=True
            )


# ============================================================
# 6. EJECUCION
# ============================================================

def main():

    df, origen = cargar_datos()

    env = simpy.Environment()

    puerto = Puerto(env, origen)

    env.process(
        llegadas(env, puerto, df)
    )

    print(
        "Iniciando simulacion con turnos...",
        flush=True
    )

    env.run(
        until=float(df["hora_sim"].max()) + 1000
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    r = pd.DataFrame(puerto.registros)

    if len(r) != len(df):

        raise RuntimeError(
            f"Se atendieron {len(r)} de {len(df)} escalas"
        )

    r = r.sort_values("fecha_hora_arribo")

    r.to_csv(
        SALIDA / "resultados_por_buque.csv",
        index=False
    )

    horizonte = max(
        float(df["hora_sim"].max()),
        float(r["hora_salida_sim"].max())
    )

    resumen = {

        "escalas_atendidas": len(r),

        "espera_media_rada_h": (
            r["espera_rada_h"].mean()
        ),

        "operacion_media_transcurrida_h": (
            r["operacion_transcurrida_h"].mean()
        ),

        "estadia_media_h": (
            r["estadia_total_h"].mean()
        ),

        "cumplimiento_16h_pct": (
            100 * (r["estadia_total_h"] <= 16).mean()
        ),

        "utilizacion_bermas_pct": (
            100
            * puerto.uso_bermas_h
            / (BERMAS * horizonte)
        ),

        "utilizacion_sts_pct": (
            100
            * puerto.uso_gruas_h
            / (STS * horizonte)
        ),

        "activaciones_standby": puerto.activaciones

    }

    pd.DataFrame([resumen]).to_csv(
        SALIDA / "indicadores_resumen.csv",
        index=False
    )

    print("\n=== CODIGO 6: SIMULADOR CON TURNOS ===")

    for k, v in resumen.items():

        if isinstance(v, (float, np.floating)):

            print(f"{k}: {v:.3f}")

        else:

            print(f"{k}: {v}")

    print("\nResultados guardados en:", SALIDA)

    print(
        "Modelo intermedio: una STS por buque. "
        "Pendiente incorporar asignacion de 2 STS, "
        "patio, tractores y validacion."
    )


if __name__ == "__main__":

    main()