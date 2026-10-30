# TP2b Puerto Sur - Grupo 11 | Codigo 3: analisis de movimientos
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CARPETA = BASE / 'Graficos_Movimientos'
CARPETA.mkdir(parents=True, exist_ok=True)
DATOS = BASE / 'PuertoSur_escalas_G11.csv'
TIPOS = ['mov_dry', 'mov_reefer', 'mov_oog']


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(CARPETA / nombre, dpi=200, bbox_inches='tight')
    plt.close()
    print('Guardado:', nombre)


df = pd.read_csv(DATOS)
df['fecha_hora_arribo'] = pd.to_datetime(df['fecha_hora_arribo'], errors='raise')
for col in TIPOS:
    df[col] = pd.to_numeric(df[col], errors='raise')
    if (df[col] < 0).any():
        raise ValueError(f'Hay movimientos negativos en {col}. Revisar CSV.')

df['mov_total'] = df[TIPOS].sum(axis=1)
df['mes'] = df['fecha_hora_arribo'].dt.month
for col in TIPOS:
    df['proporcion_' + col[4:]] = np.where(df['mov_total'] > 0, df[col] / df['mov_total'], np.nan)

print('\n=== CONTROL DE DATOS ===')
print('Escalas:', len(df))
print('Duplicados id_escala:', df['id_escala'].duplicated().sum())
print('Escalas sin movimientos:', (df['mov_total'] == 0).sum())
print('Valores ausentes:', df[['servicio'] + TIPOS].isna().sum().to_dict())
print('\n=== MOVIMIENTOS POR ESCALA ===')
resumen = df[TIPOS + ['mov_total']].describe(percentiles=[0.05, 0.5, 0.95, 0.99]).T
print(resumen.round(2).to_string())
print('\n=== POR SERVICIO (PROMEDIOS POR ESCALA) ===')
servicio = df.groupby('servicio')[TIPOS + ['mov_total']].agg(['count', 'mean', 'median', 'std'])
print(servicio.round(2).to_string())
print('\n=== MEZCLA GLOBAL (MOVIMIENTOS, NO ESCALAS) ===')
sumas = df[TIPOS].sum()
print(pd.DataFrame({'movimientos': sumas, 'participacion': sumas / sumas.sum()}).round(4).to_string())
print('\n=== CORRELACIONES ENTRE MOVIMIENTOS ===')
print(df[TIPOS].corr().round(3).to_string())

# Grafico 1: distribuciones de movimientos por tipo
fig, ax = plt.subplots(figsize=(10, 5))
for col, etiqueta in zip(TIPOS, ['Dry', 'Reefer', 'OOG']):
    ax.hist(df[col], bins=35, alpha=0.45, label=etiqueta)
ax.set(title='Distribucion de movimientos por escala', xlabel='Movimientos por escala', ylabel='Cantidad de escalas')
ax.legend()
ax.grid(alpha=0.2)
guardar('01_distribucion_por_tipo.png')

# Grafico 2: movimientos totales por servicio
fig, ax = plt.subplots(figsize=(10, 5))
datos = [sub['mov_total'].values for _, sub in df.groupby('servicio')]
nombres = [nombre for nombre, _ in df.groupby('servicio')]
ax.boxplot(datos, tick_labels=nombres, showfliers=True)
ax.set(title='Movimientos totales por escala y servicio', ylabel='Movimientos por escala')
ax.grid(axis='y', alpha=0.2)
guardar('02_totales_por_servicio.png')

# Grafico 3: mix agregado por servicio
mix_serv = df.groupby('servicio')[TIPOS].sum()
mix_pct = mix_serv.div(mix_serv.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 5))
mix_pct.plot(kind='bar', stacked=True, ax=ax)
ax.set(title='Composicion de movimientos por servicio', xlabel='Servicio', ylabel='Porcentaje de movimientos (%)')
ax.tick_params(axis='x', rotation=15)
ax.legend(['Dry', 'Reefer', 'OOG'])
guardar('03_mix_por_servicio.png')

# Grafico 4: mix mensual historico (peso de reefer sobre movimientos totales)
mensual = df.groupby('mes')[TIPOS].sum().reindex(range(1, 13), fill_value=0)
participacion_reefer = mensual['mov_reefer'] / mensual.sum(axis=1).replace(0, np.nan) * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(participacion_reefer.index, participacion_reefer.values, marker='o')
ax.set(title='Participacion de reefer por mes (serie historica agregada)', xlabel='Mes', ylabel='% de movimientos reefer')
ax.set_xticks(range(1, 13))
ax.grid(alpha=0.3)
guardar('04_reefer_por_mes.png')

# Grafico 5: relacion Dry-Reefer
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df['mov_dry'], df['mov_reefer'], s=9, alpha=0.25)
ax.set(title='Relacion entre movimientos Dry y Reefer', xlabel='Movimientos Dry', ylabel='Movimientos Reefer')
ax.grid(alpha=0.2)
guardar('05_relacion_dry_reefer.png')

# Exportar tablas y muestra procesada para el siguiente codigo
resumen.to_csv(CARPETA / 'resumen_movimientos.csv', encoding='utf-8-sig')
servicio.to_csv(CARPETA / 'estadisticas_por_servicio.csv', encoding='utf-8-sig')
mix_pct.to_csv(CARPETA / 'mix_por_servicio_pct.csv', encoding='utf-8-sig')
mensual.to_csv(CARPETA / 'movimientos_mensuales.csv', encoding='utf-8-sig')
df.to_csv(BASE / 'datos_movimientos_procesados.csv', index=False, encoding='utf-8-sig')
print('\nListo. Resultados y 5 graficos guardados en:', CARPETA)
print('Archivo generado: datos_movimientos_procesados.csv')
