# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 12: ESCENARIOS DE INVERSION Y DEMANDA
#
# Requiere los codigos corregidos 8, 9 y 10 en esta carpeta.
# Ejecutar: python simulador_escenarios.py
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt

from simulador_camiones import Puerto as PuertoBase
from simulador_patio import cargar_datos, RNG as RNG_MODELO
import simulador_camiones
import simulador_patio

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "Resultados_Escenarios"
SALIDA.mkdir(parents=True, exist_ok=True)

# Configuraciones que permiten aislar ambas inversiones.
CONFIGURACIONES = [
    ("Actual", 3, 3),
    ("Cuarta_STS", 3, 4),
    ("Cuarta_berma", 4, 3),
    ("Ambas", 4, 4),
]
DEMANDAS = [
    ("Actual", 1.00),
    ("Mas_35_pct", 1.35),
    ("Mas_100_pct", 2.00),
]

# Supuesto metodologico: la demanda aumenta comprimiendo
# intervalos entre arribos, manteniendo los movimientos,
# servicios y orden de las escalas historicas.
# NO representa una proyeccion calendarizada de la demanda.
#
# Horizonte: se observan los resultados hasta 7 dias despues
# de la ultima llegada. Las escalas pendientes NO se omiten.
DIAS_EXTRA = 7
SEMILLA = 51


class PuertoEscenario(PuertoBase):
    def __init__(self, env, origen, n_bermas, n_sts):
        super().__init__(env, origen)

        self.n_bermas = n_bermas
        self.n_sts = n_sts

        # Sustituir recursos fisicos: se respetan las
        # restricciones nocturnas heredadas del modelo.
        self.bermas = simpy.Resource(env, capacity=n_bermas)
        self.gruas = simpy.Container(
            env, capacity=n_sts, init=n_sts
        )


def generador_llegadas(env, puerto, datos):
    for i, (_, fila) in enumerate(datos.iterrows(), start=1):
        espera = max(0.0, float(fila["hora_sim"]) - env.now)
        yield env.timeout(espera)
        env.process(puerto.atender(fila))
        if i % 2000 == 0:
            print(f"  Ingresaron {i} de {len(datos)} escalas", flush=True)


def ejecutar_escenario(nombre, bermas, sts, demanda, factor, datos_original, origen):
    # Misma secuencia aleatoria inicial en todos los escenarios.
    RNG_MODELO.bit_generator.state = (
        np.random.default_rng(SEMILLA).bit_generator.state
    )

    datos = datos_original.copy()
    datos["hora_sim"] = datos["hora_sim"] / factor

    env = simpy.Environment()
    puerto = PuertoEscenario(env, origen, bermas, sts)
    env.process(generador_llegadas(env, puerto, datos))

    ultima_llegada = float(datos["hora_sim"].max())
    horizonte = ultima_llegada + 24 * DIAS_EXTRA
    env.run(until=horizonte)

    # Se consideran escalas TERMINADAS dentro del horizonte.
    # Las escalas pendientes quedan reflejadas por separado.
    finalizadas = pd.DataFrame(puerto.registros)
    terminadas = len(finalizadas)
    pendientes = len(datos) - terminadas
    puerto.actualizar_patio()

    carpeta = SALIDA / f"{nombre}__{demanda}"
    carpeta.mkdir(exist_ok=True)

    if not finalizadas.empty:
        finalizadas.to_csv(carpeta / "buques_finalizados.csv", index=False)
        estadia = float(finalizadas["estadia_total_h"].mean())
        espera = float(finalizadas["espera_rada_h"].mean())
        cumple = float((finalizadas["estadia_total_h"] <= 16).mean() * 100)
    else:
        pd.DataFrame().to_csv(carpeta / "buques_finalizados.csv", index=False)
        estadia = np.nan
        espera = np.nan
        cumple = np.nan

    # Si hay buques sin finalizar, sus KPI no son comparables
    # con los de un escenario estable: hay censura temporal.
    resultado = {
        "configuracion": nombre,
        "demanda": demanda,
        "factor_demanda": factor,
        "bermas": bermas,
        "sts": sts,
        "escalas_ingresadas": len(datos),
        "escalas_finalizadas": terminadas,
        "escalas_pendientes": pendientes,
        "pendientes_pct": 100 * pendientes / len(datos),
        "estadia_media_finalizadas_h": estadia,
        "espera_media_finalizadas_h": espera,
        "finalizadas_cumplen_16h_pct": cumple,
        "utilizacion_bermas_pct": 100 * puerto.uso_bermas / (bermas * horizonte),
        "utilizacion_sts_pct": 100 * puerto.uso_gruas / (sts * horizonte),
        "ocupacion_maxima_patio_pct": 100 * puerto.ocupacion_maxima / 3000,
        "tiempo_sobre_75_pct": 100 * puerto.tiempo_congestion / horizonte,
        "horizonte_simulado_h": horizonte,
    }

    pd.DataFrame([resultado]).to_csv(
        carpeta / "indicadores.csv", index=False
    )
    return resultado


def main():
    datos, origen = cargar_datos()
    resultados = []

    # Se conservan las politicas de camiones del Codigo 10.
    simulador_camiones.POLITICA = "actual"

    for demanda, factor in DEMANDAS:
        for nombre, bermas, sts in CONFIGURACIONES:
            print(
                f"\nEscenario: {nombre} | Demanda {demanda} "
                f"| {bermas} bermas, {sts} STS",
                flush=True,
            )
            fila = ejecutar_escenario(
                nombre, bermas, sts, demanda, factor, datos, origen
            )
            resultados.append(fila)
            print(
                f"  Finalizadas: {fila['escalas_finalizadas']}/"
                f"{fila['escalas_ingresadas']} | "
                f"Pendientes: {fila['escalas_pendientes']} | "
                f"Estadia finalizadas: "
                f"{fila['estadia_media_finalizadas_h']:.2f} h",
                flush=True,
            )

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(SALIDA / "comparacion_escenarios.csv", index=False)

    # Grafico 1: estadia por escenario.
    pivot = tabla.pivot(
        index="configuracion",
        columns="demanda",
        values="estadia_media_finalizadas_h"
    ).reindex(
        index=[c[0] for c in CONFIGURACIONES],
        columns=[d[0] for d in DEMANDAS]
    )
    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=0)
    ax.axhline(16, ls="--", color="black", label="Umbral 16 h")
    ax.set_ylabel("Estadia media de escalas finalizadas (h)")
    ax.set_title("Escenarios: estadias observadas (atencion a pendientes)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SALIDA / "01_estadia_escenarios.png", dpi=180)
    plt.close()

    # Grafico 2: escalas pendientes (clave para detectar saturacion).
    pivot_p = tabla.pivot(
        index="configuracion", columns="demanda", values="pendientes_pct"
    ).reindex(
        index=[c[0] for c in CONFIGURACIONES],
        columns=[d[0] for d in DEMANDAS]
    )
    ax = pivot_p.plot(kind="bar", figsize=(11, 5), rot=0)
    ax.set_ylabel("Escalas pendientes al cierre (%)")
    ax.set_title("Escalas no terminadas: alerta de saturacion/censura")
    plt.tight_layout()
    plt.savefig(SALIDA / "02_pendientes_escenarios.png", dpi=180)
    plt.close()

    print("\n=== CODIGO 12: ESCENARIOS ===")
    columnas = [
        "configuracion", "demanda", "escalas_finalizadas",
        "escalas_pendientes", "estadia_media_finalizadas_h",
        "utilizacion_bermas_pct", "utilizacion_sts_pct"
    ]
    print(tabla[columnas].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nResultados guardados en:", SALIDA)
    print(
        "NOTA: resultados exploratorios; mayor demanda se genero "
        "comprimiendo los tiempos de arribos. "
        "Si quedan buques pendientes, NO interpretar la estadia "
        "de los finalizados como la estadia real de todos."
    )
    print(
        "La cuarta STS incrementa equipos instalados, pero "
        "no aumenta automaticamente la dotacion nocturna."
    )


if __name__ == "__main__":
    main()
