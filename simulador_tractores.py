# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 9: SIMULADOR CON TRACTORES DE PATIO
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import simpy
import matplotlib.pyplot as plt

# Reutilizamos el simulador del Codigo 8.
from simulador_patio import (
    Puerto as PuertoBase,
    cargar_datos,
    llegadas,
    CAPACIDAD_PATIO,
    BERMAS,
    STS_TOTALES,
    TASA,
    FRACCION_DESCARGA,
    LOTE_DESCARGA,
    RNG
)

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "Resultados_Tractores"
SALIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARAMETROS PROVISIONALES
# ============================================================

TRACTORES_DIA = 4
TRACTORES_NOCHE = 2

CONTENEDORES_POR_VIAJE = 5

TIEMPO_VIAJE_MIN = 15

# Los valores anteriores son SUPUESTOS para evaluar
# el funcionamiento del sistema.
# Se verificaran posteriormente mediante sensibilidad.

SEMILLA = 51


# ============================================================
# 1. PUERTO CON TRACTORES
# ============================================================

class Puerto(PuertoBase):
    def __init__(self, env, origen):
        super().__init__(env, origen)
        self.tractores = simpy.Resource(env, capacity=TRACTORES_DIA)
        self.tractores_noche = simpy.Resource(env, capacity=TRACTORES_NOCHE)
        self.esperas_tractores = []
        self.horas_tractores = 0.0
        self.viajes_realizados = 0
        self.contenedores_transportados = 0
        self.historial_tractores = []

    def _viaje(self, cantidad):
        """Viaje independiente: varios tractores pueden operar en paralelo.

        El recurso nocturno limita los viajes que comienzan de noche.
        Si el viaje cruza las 08:00, puede finalizar sin interrupcion.
        """
        inicio_espera = self.env.now
        _, _, noche = self.turno_actual()
        if noche:
            with self.tractores_noche.request() as permiso_noche:
                yield permiso_noche
                with self.tractores.request() as permiso:
                    yield permiso
                    espera = self.env.now - inicio_espera
                    self.esperas_tractores.append(espera)
                    inicio_viaje = self.env.now
                    duracion = TIEMPO_VIAJE_MIN / 60.0
                    yield self.env.timeout(duracion)
        else:
            with self.tractores.request() as permiso:
                yield permiso
                espera = self.env.now - inicio_espera
                self.esperas_tractores.append(espera)
                inicio_viaje = self.env.now
                duracion = TIEMPO_VIAJE_MIN / 60.0
                yield self.env.timeout(duracion)

        self.horas_tractores += duracion
        self.viajes_realizados += 1
        self.contenedores_transportados += cantidad
        self.historial_tractores.append({
            "hora_sim": inicio_viaje,
            "fin_viaje": self.env.now,
            "duracion_h": duracion,
            "contenedores": cantidad,
            "espera_tractor_h": espera,
        })
        # El ingreso a los slots ocurre al finalizar cada viaje,
        # no al final de todos los viajes del buque.
        return (yield from super().ingresar_contenedores(cantidad))

    def transportar_lote(self, cantidad):
        """Despacha viajes en paralelo; cada viaje ingresa sus slots al terminar."""
        if cantidad <= 0:
            return 0.0
        pendientes = int(cantidad)
        viajes = []
        while pendientes:
            carga = min(CONTENEDORES_POR_VIAJE, pendientes)
            viajes.append(self.env.process(self._viaje(carga)))
            pendientes -= carga
        valores = yield self.env.all_of(viajes)
        return max(float(valores[v] or 0.0) for v in viajes)

    def atender(self, fila):
        """Las gruas descargan lotes mientras los tractores trasladan lotes anteriores.

        El buque espera al final la finalizacion de todos sus traslados.
        Capacidad de patio y bloqueos continuan gobernados por el Codigo 8.
        """
        llegada = self.env.now
        with self.bermas.request() as permiso:
            yield permiso
            espera_rada = self.env.now - llegada
            ingreso_berma = self.env.now
            maniobras = RNG.uniform(1, 2)   # supuesto del TP
            yield self.env.timeout(maniobras / 2)

            movimientos = int(sum(int(fila[c]) for c in TASA))
            cantidad_descarga = int(round(movimientos * FRACCION_DESCARGA))
            restante = cantidad_descarga
            lotes = []
            while restante > 0:
                n = min(LOTE_DESCARGA, restante)
                lotes.append(n)
                restante -= n
            if not lotes:
                lotes = [0]

            inicio_op = self.env.now
            operacion_efectiva = 0.0
            horas_sts = 0.0
            uso_dos = False
            tareas_transporte = []

            for lote in lotes:
                fraccion = (lote / cantidad_descarga) if cantidad_descarga else 1.0
                subcarga = {c: float(fila[c]) * fraccion for c in TASA}
                activo, hs, dos = yield from self.operar(subcarga)
                operacion_efectiva += activo
                horas_sts += hs
                uso_dos = uso_dos or dos

                if lote:
                    # NO hacemos yield aqui: el proximo lote de grua
                    # comienza mientras los tractores trabajan.
                    tareas_transporte.append(
                        self.env.process(self.transportar_lote(lote))
                    )

            fin_gruas = self.env.now
            if tareas_transporte:
                demoras = yield self.env.all_of(tareas_transporte)
                # Espera posterior a gruas por la ultima operacion terrestre.
                espera_tractores_fin = max(0.0, self.env.now - fin_gruas)
                espera_patio = sum(float(demoras[t] or 0.0) for t in tareas_transporte)
            else:
                espera_tractores_fin = 0.0
                espera_patio = 0.0

            fin_op = self.env.now
            yield self.env.timeout(maniobras / 2)
            salida = self.env.now
            self.uso_bermas += salida - ingreso_berma

        self.registros.append({
            "id_escala": fila["id_escala"],
            "servicio": fila["servicio"],
            "fecha_hora_arribo": fila["fecha_hora_arribo"],
            "espera_rada_h": espera_rada,
            "espera_patio_h": espera_patio,
            "espera_tractores_fin_h": espera_tractores_fin,
            "operacion_efectiva_h": operacion_efectiva,
            "operacion_transcurrida_h": fin_op - inicio_op,
            "horas_sts": horas_sts,
            "uso_dos_gruas": uso_dos,
            "contenedores_descargados": cantidad_descarga,
            "estadia_total_h": salida - llegada,
            "hora_salida_sim": salida,
        })


# ============================================================
# 5. EJECUTAR SIMULACION
# ============================================================

def main():

    datos, origen = cargar_datos()

    env = simpy.Environment()

    puerto = Puerto(env, origen)

    env.process(
        llegadas(env, puerto, datos)
    )

    print(
        "Iniciando simulacion con tractores...",
        flush=True
    )

    horizonte_llegadas = float(
        datos["hora_sim"].max()
    )

    env.run(
        until=horizonte_llegadas + 1000
    )

    # Permitir que terminen las escalas pendientes.

    extra = 0

    while (
        len(puerto.registros) < len(datos)
        and extra < 20000
    ):

        env.run(
            until=env.now + 500
        )

        extra += 500

    if len(puerto.registros) != len(datos):

        raise RuntimeError(
            f"Se atendieron {len(puerto.registros)} "
            f"de {len(datos)} escalas. "
            "Revisar capacidad y estabilidad."
        )

    # ========================================================
    # 6. RESULTADOS
    # ========================================================

    r = pd.DataFrame(
        puerto.registros
    ).sort_values(
        "fecha_hora_arribo"
    )

    r.to_csv(
        SALIDA / "resultados_por_buque.csv",
        index=False
    )

    horizonte = max(
        horizonte_llegadas,
        float(r["hora_salida_sim"].max())
    )

    # Cerrar estadisticas del patio.

    puerto.actualizar_patio()

    resumen = {

        "escalas_atendidas": len(r),

        "espera_media_rada_h": (
            r["espera_rada_h"].mean()
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

        "ocupacion_maxima_patio_pct": (
            100
            * puerto.ocupacion_maxima
            / CAPACIDAD_PATIO
        ),

        "espera_media_tractores_fin_h": r["espera_tractores_fin_h"].mean(),

        "espera_media_tractores_h": (
            np.mean(puerto.esperas_tractores)
            if puerto.esperas_tractores
            else 0.0
        ),

        "utilizacion_tractores_pct": (
            100
            * puerto.horas_tractores
            / (TRACTORES_DIA * horizonte)
        ),

        "viajes_tractores": (
            puerto.viajes_realizados
        ),

        "contenedores_transportados": (
            puerto.contenedores_transportados
        )
    }

    pd.DataFrame([resumen]).to_csv(
        SALIDA / "indicadores_resumen.csv",
        index=False
    )

    pd.DataFrame(
        puerto.historial_tractores
    ).to_csv(
        SALIDA / "historial_tractores.csv",
        index=False
    )

    # ========================================================
    # 7. GRAFICOS
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.hist(
        puerto.esperas_tractores,
        bins=40,
        edgecolor="black"
    )

    plt.title(
        "Puerto Sur - Esperas por disponibilidad de tractores"
    )

    plt.xlabel("Espera (horas)")
    plt.ylabel("Frecuencia")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SALIDA / "01_esperas_tractores.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.hist(
        r["estadia_total_h"],
        bins=40,
        edgecolor="black"
    )

    plt.axvline(
        16,
        color="red",
        linestyle="--",
        label="Objetivo: 16 horas"
    )

    plt.title(
        "Puerto Sur - Estadias con restriccion de tractores"
    )

    plt.xlabel("Estadia (horas)")
    plt.ylabel("Cantidad de buques")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SALIDA / "02_estadias_con_tractores.png",
        dpi=200
    )

    plt.close()

    # ========================================================
    # 8. MOSTRAR RESULTADOS
    # ========================================================

    print(
        "\n=== CODIGO 9: SIMULADOR CON TRACTORES ==="
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
        "\nModelo provisional con gruas y tractores simultaneos: pendientes camiones, "
        "validacion de cantidad de tractores y "
        "calibracion de tiempos y capacidades de transporte."
    )


if __name__ == "__main__":
    main()