# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 8: SIMULADOR CON PATIO DE CONTENEDORES
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import simpy

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "PuertoSur_escalas_G11.csv"
SALIDA = BASE / "Resultados_Patio"
SALIDA.mkdir(exist_ok=True)

# ============================================================
# PARAMETROS
# ============================================================

BERMAS = 3
STS_TOTALES = 3
MAX_STS_BUQUE = 2

INTERFERENCIA = 0.85

TASA = {
    "mov_dry": 28.0,
    "mov_reefer": 21.0,
    "mov_oog": 12.0
}

# PATIO

CAPACIDAD_PATIO = 3000
UMBRAL_CONGESTION = 0.75

PERMANENCIA_MEDIA_H = 72.0

# SUPUESTOS PROVISIONALES:
# - Ocupacion inicial del patio: 1500 slots.
# - 50% de movimientos corresponden a descargas.
# - Cada contenedor descargado ocupa un slot.
# - Permanencia deterministica de 72 h.
# - No se representan todavia camiones ni tractores.
# Estos supuestos se revisaran en los codigos siguientes.

OCUPACION_INICIAL = 1500
FRACCION_DESCARGA = 0.50
LOTE_DESCARGA = 10  # max contenedores por lote agregado de descarga

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

    if (
        df[["mov_dry", "mov_reefer", "mov_oog"]] < 0
    ).any().any():
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

        # El contenedor representa slots disponibles.

        self.slots_libres = simpy.Container(
            env,
            capacity=CAPACIDAD_PATIO,
            init=CAPACIDAD_PATIO - OCUPACION_INICIAL
        )

        self.ocupacion = OCUPACION_INICIAL
        # Stock al iniciar debe salir: no puede ocupar slots para siempre.
        # Hipotesis provisional: edades residuales distribuidas en 72 horas.
        env.process(self.liberar_stock_inicial())

        self.registros = []

        self.uso_bermas = 0.0
        self.uso_gruas = 0.0

        self.relevos = {}
        self.recursos_noche = {}

        self.activaciones_standby = 0

        # Indicadores del patio

        self.ocupacion_maxima = OCUPACION_INICIAL
        self.area_ocupacion = 0.0

        self.tiempo_congestion = 0.0
        self.ultimo_cambio_patio = 0.0

        self.esperas_patio = []

        self.contenedores_ingresados = 0
        self.contenedores_retirados = 0

        self.historial_patio = [
            {
                "hora_sim": 0.0,
                "ocupacion": OCUPACION_INICIAL,
                "ocupacion_pct": (
                    100 * OCUPACION_INICIAL / CAPACIDAD_PATIO
                )
            }
        ]

    # ========================================================
    # 3. CONTROL DEL TIEMPO Y TURNOS
    # ========================================================

    def calendario(self):

        return self.origen_h + self.env.now

    def turno_actual(self):

        t = self.calendario()

        dia = int(t // 24)
        hora = t % 24

        inicio = dia * 24

        if hora < 8:
            return inicio, inicio + 8, True

        if hora < 16:
            return inicio + 8, inicio + 16, False

        return inicio + 16, inicio + 24, False

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

    # ========================================================
    # 4. DISPONIBILIDAD NOCTURNA
    # ========================================================

    def recurso_nocturno(self, inicio):

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
    # 5. ESTADISTICAS DEL PATIO
    # ========================================================

    def actualizar_patio(self):

        ahora = self.env.now

        delta = ahora - self.ultimo_cambio_patio

        self.area_ocupacion += (
            self.ocupacion * delta
        )

        if (
            self.ocupacion / CAPACIDAD_PATIO
            > UMBRAL_CONGESTION
        ):

            self.tiempo_congestion += delta

        self.ultimo_cambio_patio = ahora

    def registrar_patio(self):

        self.ocupacion_maxima = max(
            self.ocupacion_maxima,
            self.ocupacion
        )

        self.historial_patio.append({

            "hora_sim": self.env.now,

            "ocupacion": self.ocupacion,

            "ocupacion_pct": (
                100 * self.ocupacion / CAPACIDAD_PATIO
            )

        })

    # ========================================================
    # 6. INGRESO DE CONTENEDORES AL PATIO
    # ========================================================

    def liberar_stock_inicial(self):
        # Se libera gradualmente el inventario inicial en las primeras 72 h.
        # No atribuir estos retiros a buques simulados.
        n_periodos = 72
        base, sobrante = divmod(OCUPACION_INICIAL, n_periodos)
        for i in range(n_periodos):
            yield self.env.timeout(PERMANENCIA_MEDIA_H / n_periodos)
            cantidad = base + (1 if i < sobrante else 0)
            self.actualizar_patio()
            self.ocupacion -= cantidad
            yield self.slots_libres.put(cantidad)
            self.registrar_patio()

    def ingresar_contenedores(self, cantidad):
        # Solicitar sólo un lote de descarga; NO un buque entero.
        # Si no quedan slots, la grúa queda bloqueada hasta liberación.
        if cantidad <= 0:
            return 0.0
        inicio = self.env.now
        yield self.slots_libres.get(cantidad)
        demora = self.env.now - inicio
        self.esperas_patio.append(demora)
        self.actualizar_patio()
        self.ocupacion += cantidad
        self.contenedores_ingresados += cantidad
        assert 0 <= self.ocupacion <= CAPACIDAD_PATIO
        self.registrar_patio()
        self.env.process(self.liberar_contenedores(cantidad))
        return demora

    def liberar_contenedores(self, cantidad):
        yield self.env.timeout(PERMANENCIA_MEDIA_H)
        self.actualizar_patio()
        self.ocupacion -= cantidad
        self.contenedores_retirados += cantidad
        yield self.slots_libres.put(cantidad)
        self.registrar_patio()

    # ========================================================
    # 8. OPERACION DE GRUAS
    # ========================================================

    def operar(self, fila):

        trabajo_restante = sum(
            float(fila[k]) / tasa
            for k, tasa in TASA.items()
        )

        tiempo_efectivo = 0.0
        horas_sts = 0.0

        uso_dos = False

        while trabajo_restante > 1e-8:

            yield from self.esperar_relevo()

            inicio, fin, noche = self.turno_actual()

            recurso_noche = None

            if noche:

                recurso_noche = (
                    self.recurso_nocturno(inicio)
                )

                yield recurso_noche.get(1)

            yield self.gruas.get(1)

            n_gruas = 1

            if self.gruas.level >= 1:

                if noche:

                    if recurso_noche.level >= 1:

                        yield recurso_noche.get(1)

                        yield self.gruas.get(1)

                        n_gruas = 2

                else:

                    yield self.gruas.get(1)

                    n_gruas = 2

            factor = (
                INTERFERENCIA
                if n_gruas == 2
                else 1.0
            )

            if n_gruas == 2:
                uso_dos = True

            productividad = n_gruas * factor

            tiempo_disponible = (
                fin - self.calendario()
            )

            duracion = min(
                trabajo_restante / productividad,
                max(0.0, tiempo_disponible)
            )

            if duracion > 1e-9:

                yield self.env.timeout(duracion)

                trabajo_restante -= (
                    duracion * productividad
                )

                tiempo_efectivo += duracion

                horas_sts += n_gruas * duracion

                self.uso_gruas += (
                    n_gruas * duracion
                )

            yield self.gruas.put(n_gruas)

            if noche:
                yield recurso_noche.put(n_gruas)

            if duracion <= 1e-9:
                yield self.env.timeout(1e-6)

        return (
            tiempo_efectivo,
            horas_sts,
            uso_dos
        )

    # ========================================================
    # 9. ATENCION DE BUQUES
    # ========================================================

    def atender(self, fila):

        llegada = self.env.now

        with self.bermas.request() as permiso:

            yield permiso

            espera_rada = self.env.now - llegada

            ingreso_berma = self.env.now

            maniobras = RNG.uniform(1, 2)

            # Atraque

            yield self.env.timeout(
                maniobras / 2
            )

            # Cantidad de contenedores descargados

            movimientos = int(
                fila["mov_dry"]
                + fila["mov_reefer"]
                + fila["mov_oog"]
            )

            cantidad_descarga = int(
                round(movimientos * FRACCION_DESCARGA)
            )

            # Carga/descarga progresiva: cada lote usa STS y luego
            # ingresa al patio. El patio puede bloquear el siguiente lote.
            # NO se reservan todos los slots antes de iniciar la operación.
            inicio_op = self.env.now
            operacion_efectiva = 0.0
            horas_sts = 0.0
            uso_dos = False
            espera_patio = 0.0
            restantes = cantidad_descarga
            movimientos_totales = float(movimientos)
            # Partir el trabajo total por lotes de descarga; mantener la
            # mezcla del buque proporcional en cada tramo.
            lotes = []
            if cantidad_descarga > 0:
                while restantes > 0:
                    lote = min(LOTE_DESCARGA, restantes)
                    lotes.append(lote)
                    restantes -= lote
            else:
                lotes = [0]

            for j, lote in enumerate(lotes):
                if cantidad_descarga > 0:
                    fraccion = lote / cantidad_descarga
                else:
                    fraccion = 1.0
                subcarga = {
                    k: float(fila[k]) * fraccion
                    for k in TASA
                }
                (activo, hs, dos) = yield from self.operar(subcarga)
                operacion_efectiva += activo
                horas_sts += hs
                uso_dos = uso_dos or dos
                if lote:
                    espera_patio += (
                        yield from self.ingresar_contenedores(lote)
                    )

            fin_op = self.env.now

            # Desatraque

            yield self.env.timeout(
                maniobras / 2
            )

            salida = self.env.now

            self.uso_bermas += (
                salida - ingreso_berma
            )

        self.registros.append({

            "id_escala": fila["id_escala"],

            "servicio": fila["servicio"],

            "fecha_hora_arribo": fila["fecha_hora_arribo"],

            "espera_rada_h": espera_rada,

            "espera_patio_h": espera_patio,

            "operacion_efectiva_h": operacion_efectiva,

            "operacion_transcurrida_h": (
                fin_op - inicio_op
            ),

            "horas_sts": horas_sts,

            "uso_dos_gruas": uso_dos,

            "contenedores_descargados": cantidad_descarga,

            "estadia_total_h": salida - llegada,

            "hora_salida_sim": salida

        })


# ============================================================
# 10. GENERADOR DE LLEGADAS
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
# 11. EJECUCION PRINCIPAL
# ============================================================

def main():

    df, origen = cargar_datos()

    env = simpy.Environment()

    puerto = Puerto(env, origen)

    env.process(
        llegadas(env, puerto, df)
    )

    print(
        "Iniciando simulacion con patio...",
        flush=True
    )

    # Se permite finalizar la atencion de los buques
    # despues de la ultima llegada historica.

    # Ejecutar al menos hasta finalizar las llegadas y todos los buques.
    # Las liberaciones futuras del patio pueden permanecer programadas.
    env.run(until=float(df["hora_sim"].max()) + 1000)
    if len(puerto.registros) != len(df):
        # Identificar si se mantiene congestión en lugar de ocultarla.
        extra = 0
        while len(puerto.registros) < len(df) and extra < 20000:
            env.run(until=env.now + 500)
            extra += 500
        if len(puerto.registros) < len(df):
            raise RuntimeError(
                f"Pendientes: {len(df)-len(puerto.registros)}; "
                "revisar estabilidad y restricciones del patio"
            )

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

    # Actualizar las estadisticas del patio
    # hasta el horizonte de observacion.

    puerto.actualizar_patio()

    resumen = {

        "escalas_atendidas": len(r),

        "espera_media_rada_h": (
            r["espera_rada_h"].mean()
        ),

        "espera_media_patio_h": (
            r["espera_patio_h"].mean()
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

        "ocupacion_media_patio_pct": (
            100
            * puerto.area_ocupacion
            / (CAPACIDAD_PATIO * env.now)
        ),

        "ocupacion_maxima_patio_pct": (
            100
            * puerto.ocupacion_maxima
            / CAPACIDAD_PATIO
        ),

        "tiempo_congestion_patio_pct": (
            100
            * puerto.tiempo_congestion
            / env.now
        ),

        "contenedores_ingresados": (
            puerto.contenedores_ingresados
        ),

        "contenedores_retirados": (
            puerto.contenedores_retirados
        ),

        "activaciones_standby": (
            puerto.activaciones_standby
        )

    }

    pd.DataFrame([resumen]).to_csv(
        SALIDA / "indicadores_resumen.csv",
        index=False
    )

    pd.DataFrame(
        puerto.historial_patio
    ).to_csv(
        SALIDA / "historial_ocupacion_patio.csv",
        index=False
    )

    # ========================================================
    # 12. GRAFICOS
    # ========================================================

    import matplotlib.pyplot as plt

    historial = pd.DataFrame(
        puerto.historial_patio
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        historial["hora_sim"] / 24,
        historial["ocupacion_pct"],
        linewidth=0.8
    )

    plt.axhline(
        75,
        color="orange",
        linestyle="--",
        label="Umbral de congestion (75%)"
    )

    plt.axhline(
        100,
        color="red",
        linestyle="--",
        label="Capacidad maxima (100%)"
    )

    plt.xlabel("Dia de simulacion")
    plt.ylabel("Ocupacion del patio (%)")

    plt.title(
        "Puerto Sur - Evolucion de ocupacion del patio"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SALIDA / "01_ocupacion_patio.png",
        dpi=200
    )

    plt.close()

    # Esperas por capacidad del patio

    plt.figure(figsize=(10, 5))

    plt.hist(
        r["espera_patio_h"],
        bins=40,
        edgecolor="black"
    )

    plt.xlabel("Espera por patio (horas)")
    plt.ylabel("Cantidad de buques")

    plt.title(
        "Puerto Sur - Distribucion de esperas por patio"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SALIDA / "02_esperas_patio.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # 13. RESULTADOS
    # ========================================================

    print(
        "\n=== CODIGO 8: SIMULADOR CON PATIO ==="
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
        "\nModelo provisional de patio: "
        "pendiente incorporar tractores, "
        "camiones y calibrar permanencias."
    )


if __name__ == "__main__":
    main()