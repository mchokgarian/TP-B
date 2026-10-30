# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 7: ASIGNACION DE HASTA 2 GRUAS STS
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import simpy

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "PuertoSur_escalas_G11.csv"
SALIDA = BASE / "Resultados_Dos_Gruas"
SALIDA.mkdir(exist_ok=True)

# PARAMETROS

BERMAS = 3
STS_TOTALES = 3
MAX_STS_BUQUE = 2
INTERFERENCIA = 0.85

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
        raise ValueError("Existen IDs duplicados")

    if (df[["mov_dry", "mov_reefer", "mov_oog"]] < 0).any().any():
        raise ValueError("Existen movimientos negativos")

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
            origen.hour
            + origen.minute / 60
            + origen.second / 3600
        )

        self.bermas = simpy.Resource(
            env,
            capacity=BERMAS
        )

        self.gruas = simpy.Container(
            env,
            capacity=STS_TOTALES,
            init=STS_TOTALES
        )

        self.registros = []

        self.uso_bermas = 0.0
        self.uso_gruas = 0.0

        self.relevos = {}
        self.recursos_noche = {}
        self.activaciones_standby = 0

        self.buques_una_grua = 0
        self.buques_dos_gruas = 0

    # --------------------------------------------------------
    # CONTROL DE TIEMPO
    # --------------------------------------------------------

    def calendario(self):

        return self.origen_h + self.env.now

    def turno_actual(self):

        t = self.calendario()

        dia = int(t // 24)
        hora = t % 24

        inicio_dia = dia * 24

        if hora < 8:
            return inicio_dia, inicio_dia + 8, True

        if hora < 16:
            return inicio_dia + 8, inicio_dia + 16, False

        return inicio_dia + 16, inicio_dia + 24, False

    # --------------------------------------------------------
    # CAMBIOS DE TURNO
    # --------------------------------------------------------

    def duracion_relevo(self, inicio):

        if inicio not in self.relevos:

            self.relevos[inicio] = (
                RNG.uniform(10, 15) / 60
            )

        return self.relevos[inicio]

    def esperar_relevo(self):

        inicio, _, _ = self.turno_actual()

        demora = (
            inicio
            + self.duracion_relevo(inicio)
            - self.calendario()
        )

        if demora > 1e-9:
            yield self.env.timeout(demora)

    # --------------------------------------------------------
    # DISPONIBILIDAD NOCTURNA
    # --------------------------------------------------------

    def obtener_recurso_nocturno(self, inicio):

        if inicio not in self.recursos_noche:

            recurso = simpy.Container(
                self.env,
                capacity=2,
                init=1
            )

            self.recursos_noche[inicio] = recurso

            self.activaciones_standby += 1

            demora = RNG.uniform(15, 20) / 60

            self.env.process(
                self.activar_standby(recurso, demora)
            )

        return self.recursos_noche[inicio]

    def activar_standby(self, recurso, demora):

        yield self.env.timeout(demora)

        yield recurso.put(1)

    # ========================================================
    # 3. OPERACION CON HASTA DOS STS
    # ========================================================

    def operar(self, fila):

        trabajo_restante = sum(
            float(fila[k]) / tasa
            for k, tasa in TASA.items()
        )

        tiempo_efectivo = 0.0
        horas_sts_buque = 0.0

        uso_una = False
        uso_dos = False

        while trabajo_restante > 1e-8:

            yield from self.esperar_relevo()

            inicio, fin, noche = self.turno_actual()

            recurso_noche = None

            # -----------------------------------------------
            # ASIGNACION DE GRUAS
            # -----------------------------------------------

            if noche:

                recurso_noche = (
                    self.obtener_recurso_nocturno(inicio)
                )

                yield recurso_noche.get(1)

            yield self.gruas.get(1)

            n_gruas = 1

            # Se asigna una segunda STS solamente
            # si existe disponibilidad inmediata.

            if self.gruas.level >= 1:

                if noche:

                    if recurso_noche.level >= 1:

                        yield recurso_noche.get(1)

                        yield self.gruas.get(1)

                        n_gruas = 2

                else:

                    yield self.gruas.get(1)

                    n_gruas = 2

            # -----------------------------------------------
            # PRODUCTIVIDAD
            # -----------------------------------------------

            if n_gruas == 2:

                factor = INTERFERENCIA

                uso_dos = True

            else:

                factor = 1.0

                uso_una = True

            productividad = n_gruas * factor

            tiempo_disponible = (
                fin - self.calendario()
            )

            duracion = min(
                trabajo_restante / productividad,
                tiempo_disponible
            )

            if duracion > 1e-9:

                yield self.env.timeout(duracion)

                trabajo_restante -= (
                    duracion * productividad
                )

                tiempo_efectivo += duracion

                horas_sts_buque += (
                    n_gruas * duracion
                )

                self.uso_gruas += (
                    n_gruas * duracion
                )

            # -----------------------------------------------
            # LIBERAR RECURSOS
            # -----------------------------------------------

            yield self.gruas.put(n_gruas)

            if noche:

                yield recurso_noche.put(n_gruas)

            # Al finalizar el tramo, las gruas quedan
            # disponibles para una nueva asignacion.

        if uso_dos:
            self.buques_dos_gruas += 1
        else:
            self.buques_una_grua += 1

        return (
            tiempo_efectivo,
            horas_sts_buque,
            uso_dos
        )

    # ========================================================
    # 4. ATENCION DE BUQUES
    # ========================================================

    def atender(self, fila):

        llegada = self.env.now

        with self.bermas.request() as permiso:

            yield permiso

            espera = self.env.now - llegada

            ingreso = self.env.now

            maniobras = RNG.uniform(1, 2)

            # Atraque

            yield self.env.timeout(
                maniobras / 2
            )

            # Operacion

            inicio_op = self.env.now

            (
                operacion_efectiva,
                horas_sts,
                uso_dos
            ) = yield from self.operar(fila)

            fin_op = self.env.now

            # Desatraque

            yield self.env.timeout(
                maniobras / 2
            )

            salida = self.env.now

            self.uso_bermas += (
                salida - ingreso
            )

        self.registros.append({

            "id_escala": fila["id_escala"],

            "servicio": fila["servicio"],

            "fecha_hora_arribo": fila["fecha_hora_arribo"],

            "espera_rada_h": espera,

            "maniobras_h": maniobras,

            "operacion_efectiva_h": (
                operacion_efectiva
            ),

            "operacion_transcurrida_h": (
                fin_op - inicio_op
            ),

            "horas_sts": horas_sts,

            "uso_dos_gruas": uso_dos,

            "estadia_total_h": (
                salida - llegada
            ),

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
# 6. EJECUCION PRINCIPAL
# ============================================================

def main():

    df, origen = cargar_datos()

    env = simpy.Environment()

    puerto = Puerto(env, origen)

    env.process(
        llegadas(env, puerto, df)
    )

    print(
        "Iniciando simulacion con hasta 2 STS...",
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

    r = r.sort_values(
        "fecha_hora_arribo"
    )

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

        "operacion_media_h": (
            r["operacion_transcurrida_h"].mean()
        ),

        "estadia_media_h": (
            r["estadia_total_h"].mean()
        ),

        "cumplimiento_16h_pct": (
            100 * (
                r["estadia_total_h"] <= 16
            ).mean()
        ),

        "utilizacion_bermas_pct": (
            100
            * puerto.uso_bermas
            / (BERMAS * horizonte)
        ),

        "utilizacion_sts_pct": (
            100
            * puerto.uso_gruas
            / (STS_TOTALES * horizonte)
        ),

        "buques_con_una_sts": (
            puerto.buques_una_grua
        ),

        "buques_con_dos_sts": (
            puerto.buques_dos_gruas
        ),

        "activaciones_standby": (
            puerto.activaciones_standby
        )

    }

    pd.DataFrame([resumen]).to_csv(
        SALIDA / "indicadores_resumen.csv",
        index=False
    )

    # --------------------------------------------------------
    # MOSTRAR RESULTADOS
    # --------------------------------------------------------

    print(
        "\n=== CODIGO 7: SIMULADOR CON DOS STS ==="
    )

    for k, v in resumen.items():

        if isinstance(v, (float, np.floating)):

            print(f"{k}: {v:.3f}")

        else:

            print(f"{k}: {v}")

    print(
        "\nResultados guardados en:",
        SALIDA
    )

    print(
        "\nModelo intermedio: asignacion dinamica "
        "de hasta 2 STS con interferencia del 15%."
    )

    print(
        "Pendiente: validar la politica de asignacion, "
        "incorporar patio, tractores y camiones."
    )


if __name__ == "__main__":

    main()