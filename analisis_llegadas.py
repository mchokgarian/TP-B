# =====================================================
# TP2b - PUERTO SUR - GRUPO 11
# CÓDIGO 1: ANÁLISIS DE LLEGADAS DE BUQUES
# =====================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 0. CREAR CARPETA PARA LOS GRÁFICOS
# -----------------------------------------------------

BASE = Path(__file__).resolve().parent
CSV = BASE / "PuertoSur_escalas_G11.csv"

CARPETA_GRAFICOS = BASE / "Graficos_Llegadas"
CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)


def guardar_grafico(nombre):
    ruta = CARPETA_GRAFICOS / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado: {ruta}")


# -----------------------------------------------------
# 1. IMPORTAR DATOS
# -----------------------------------------------------

df = pd.read_csv(CSV)

df["fecha_hora_arribo"] = pd.to_datetime(
    df["fecha_hora_arribo"]
)

df = df.sort_values("fecha_hora_arribo").reset_index(drop=True)

print("\n===== INFORMACIÓN GENERAL =====")

print("Cantidad de escalas:", len(df))
print("Primera llegada:", df["fecha_hora_arribo"].min())
print("Última llegada:", df["fecha_hora_arribo"].max())

print("\nCantidad de escalas por servicio:")
print(df["servicio"].value_counts())


# -----------------------------------------------------
# 2. LLEGADAS POR DÍA
# -----------------------------------------------------

df["fecha"] = df["fecha_hora_arribo"].dt.date

fechas = pd.date_range(
    df["fecha_hora_arribo"].min().normalize(),
    df["fecha_hora_arribo"].max().normalize(),
    freq="D"
)

llegadas_dia = (
    df.groupby("fecha")
    .size()
    .reindex(fechas.date, fill_value=0)
)

print("\n===== LLEGADAS DIARIAS =====")

print("Promedio:", llegadas_dia.mean())
print("Desvío estándar:", llegadas_dia.std())
print("Máximo:", llegadas_dia.max())
print("Mínimo:", llegadas_dia.min())

plt.figure(figsize=(10, 5))

plt.hist(
    llegadas_dia,
    bins=np.arange(
        llegadas_dia.min() - 0.5,
        llegadas_dia.max() + 1.5,
        1
    ),
    edgecolor="black"
)

plt.title("Distribución de llegadas diarias")
plt.xlabel("Cantidad de buques por día")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)

guardar_grafico("01_llegadas_diarias.png")


# -----------------------------------------------------
# 3. LLEGADAS POR HORA
# -----------------------------------------------------

df["hora"] = df["fecha_hora_arribo"].dt.hour

llegadas_hora = (
    df.groupby("hora")
    .size()
    .reindex(range(24), fill_value=0)
)

print("\n===== LLEGADAS POR HORA =====")
print(llegadas_hora)

plt.figure(figsize=(10, 5))

plt.bar(
    llegadas_hora.index,
    llegadas_hora.values
)

plt.title("Llegadas de buques por hora del día")
plt.xlabel("Hora")
plt.ylabel("Cantidad total de llegadas")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)

guardar_grafico("02_llegadas_por_hora.png")


# -----------------------------------------------------
# 4. ESTACIONALIDAD MENSUAL
# -----------------------------------------------------

df["mes"] = df["fecha_hora_arribo"].dt.month

llegadas_mes = (
    df.groupby("mes")
    .size()
    .reindex(range(1, 13), fill_value=0)
)

print("\n===== LLEGADAS POR MES =====")
print(llegadas_mes)

plt.figure(figsize=(10, 5))

plt.bar(
    llegadas_mes.index,
    llegadas_mes.values
)

plt.title("Llegadas de buques por mes")
plt.xlabel("Mes")
plt.ylabel("Cantidad total de llegadas")
plt.xticks(range(1, 13))
plt.grid(axis="y", alpha=0.3)

guardar_grafico("03_llegadas_mensuales.png")


# -----------------------------------------------------
# 5. TIEMPOS ENTRE LLEGADAS
# -----------------------------------------------------

df["tiempo_entre_arribos"] = (
    df["fecha_hora_arribo"]
    .diff()
    .dt.total_seconds() / 3600
)

tiempos = df["tiempo_entre_arribos"].dropna()

print("\n===== TIEMPOS ENTRE ARRIBOS =====")

print("Media (horas):", tiempos.mean())
print("Mediana (horas):", tiempos.median())
print("Desvío estándar:", tiempos.std())
print("Mínimo (horas):", tiempos.min())
print("Máximo (horas):", tiempos.max())

plt.figure(figsize=(10, 5))

plt.hist(
    tiempos,
    bins=40,
    density=True,
    edgecolor="black"
)

plt.title("Distribución de tiempos entre arribos")
plt.xlabel("Tiempo entre arribos (horas)")
plt.ylabel("Densidad")
plt.grid(alpha=0.3)

guardar_grafico("04_tiempos_entre_arribos.png")


# -----------------------------------------------------
# 6. GUARDAR RESULTADOS
# -----------------------------------------------------

df.to_csv(
    BASE / "datos_llegadas_procesados.csv",
    index=False
)

print("\nAnálisis finalizado correctamente.")
print("Archivo generado: datos_llegadas_procesados.csv")
print("Carpeta de gráficos:", CARPETA_GRAFICOS)