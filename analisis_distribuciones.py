# TP2b Puerto Sur - Grupo 11 | Código 2: diagnóstico de llegadas
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / 'PuertoSur_escalas_G11.csv'
SALIDA = BASE / 'Graficos_Distribuciones'
SALIDA.mkdir(exist_ok=True)

if not ARCHIVO.exists():
    raise FileNotFoundError(f'No se encontró {ARCHIVO}. Guardá el CSV junto al programa.')

df = pd.read_csv(ARCHIVO, parse_dates=['fecha_hora_arribo'])
df = df.dropna(subset=['fecha_hora_arribo']).sort_values('fecha_hora_arribo').reset_index(drop=True)
df['anterior'] = df['fecha_hora_arribo'].shift()
df['intervalo_h'] = df['fecha_hora_arribo'].diff().dt.total_seconds() / 3600
intervalos = df['intervalo_h'].dropna()
positivos = intervalos[intervalos > 0].to_numpy()
ceros = int((intervalos == 0).sum())

print('\n=== CONTROL DE INTERVALOS ===')
print(f'Escalas: {len(df)} | Intervalos: {len(intervalos)} | Intervalos cero: {ceros}')
print(f'Media: {intervalos.mean():.3f} h | Mediana: {intervalos.median():.3f} h')
print(f'P95: {intervalos.quantile(.95):.3f} h | P99: {intervalos.quantile(.99):.3f} h')
print('\nLos 15 intervalos más largos (no se eliminan):')
extremos = df.nlargest(15, 'intervalo_h')[['anterior', 'fecha_hora_arribo', 'intervalo_h']]
print(extremos.to_string(index=False))
extremos.to_csv(SALIDA / 'intervalos_mas_largos.csv', index=False)

# Ajustes sobre intervalos estrictamente positivos: las distribuciones son continuas.
# Son ajustes exploratorios. K-S con parámetros estimados NO tiene p-valor estándar válido.
modelos = {
    'Exponencial': (stats.expon, stats.expon.fit(positivos, floc=0)),
    'Gamma': (stats.gamma, stats.gamma.fit(positivos, floc=0)),
    'Weibull': (stats.weibull_min, stats.weibull_min.fit(positivos, floc=0)),
}
filas = []
for nombre, (dist, pars) in modelos.items():
    estadistico_D = stats.kstest(positivos, lambda x: dist.cdf(x, *pars)).statistic
    logL = np.sum(dist.logpdf(positivos, *pars))
    k = len(pars) - 1  # loc fijada en cero
    aic = 2 * k - 2 * logL
    filas.append({'modelo': nombre, 'KS_D_descriptivo': estadistico_D, 'AIC': aic, 'parametros': str(pars)})
resumen = pd.DataFrame(filas).sort_values('AIC')
print('\n=== AJUSTES EXPLORATORIOS A INTERVALOS POSITIVOS ===')
print(resumen.to_string(index=False))
print('NOTA: D de K-S es descriptivo; sin p-valores porque estimamos parámetros con estos mismos datos.')
resumen.to_csv(SALIDA / 'ajustes_exploratorios.csv', index=False)

# Gráfico 1: distribución completa en escala log1p para mostrar cola extrema.
plt.figure(figsize=(10, 5))
plt.hist(np.log1p(intervalos), bins=55, edgecolor='black')
plt.xlabel('log(1 + horas entre arribos)')
plt.ylabel('Frecuencia')
plt.title('Intervalos entre arribos: distribución completa (escala transformada)')
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(SALIDA / '01_intervalos_completos.png', dpi=200)
plt.close()

# Gráfico 2: ajuste en la zona habitual. Se recorta SOLO la visualización, no el ajuste.
limite = float(np.quantile(positivos, .99))
x = np.linspace(0, limite, 350)
plt.figure(figsize=(10, 5))
plt.hist(positivos[positivos <= limite], bins=45, density=True, alpha=.45, edgecolor='black', label='Datos observados (hasta P99)')
for nombre, (dist, pars) in modelos.items():
    plt.plot(x, dist.pdf(x, *pars), label=nombre)
plt.xlabel('Horas entre arribos')
plt.ylabel('Densidad')
plt.title('Comparación exploratoria de distribuciones (vista hasta P99)')
plt.legend()
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(SALIDA / '02_ajuste_distribuciones.png', dpi=200)
plt.close()

# Gráfico 3: tasa de llegadas por hora observada (promedio por día del período).
dias = pd.date_range(df['fecha_hora_arribo'].min().normalize(), df['fecha_hora_arribo'].max().normalize(), freq='D')
tasas_hora = df['fecha_hora_arribo'].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index() / len(dias)
plt.figure(figsize=(10, 5))
plt.bar(tasas_hora.index, tasas_hora.values)
plt.xticks(range(24))
plt.xlabel('Hora del día')
plt.ylabel('Buques por hora, promedio de los días observados')
plt.title('Perfil horario de llegadas (tasas empíricas)')
plt.grid(axis='y', alpha=.25)
plt.tight_layout()
plt.savefig(SALIDA / '03_tasa_horaria.png', dpi=200)
plt.close()

# Gráfico 4: tasas mensuales normalizadas por días observados por mes.
fechas = pd.Series(dias)
dias_mes = fechas.dt.month.value_counts().reindex(range(1,13), fill_value=0)
llegadas_mes = df['fecha_hora_arribo'].dt.month.value_counts().reindex(range(1,13), fill_value=0)
tasa_mes = llegadas_mes / dias_mes
plt.figure(figsize=(10, 5))
plt.bar(range(1,13), tasa_mes)
plt.xticks(range(1,13))
plt.xlabel('Mes')
plt.ylabel('Buques por día observado')
plt.title('Tasa mensual de llegadas (normalizada por días)')
plt.grid(axis='y', alpha=.25)
plt.tight_layout()
plt.savefig(SALIDA / '04_tasa_mensual.png', dpi=200)
plt.close()

# Gráfico 5: comparación por servicio para diagnosticar heterogeneidad.
por_servicio = pd.crosstab(df['fecha_hora_arribo'].dt.hour, df['servicio']).reindex(range(24), fill_value=0) / len(dias)
ax = por_servicio.plot(figsize=(10, 5), marker='o')
ax.set_xlabel('Hora del día')
ax.set_ylabel('Llegadas por hora por día observado')
ax.set_title('Perfil horario por servicio')
ax.set_xticks(range(24))
ax.grid(alpha=.25)
plt.tight_layout()
plt.savefig(SALIDA / '05_tasas_por_servicio.png', dpi=200)
plt.close()

print(f'\nListo. Gráficos, tabla de ajustes e intervalos extremos guardados en:\n{SALIDA}')
print('IMPORTANTE: no elegimos aún un generador de llegadas. Revisaremos estabilidad temporal y adecuación del proceso.')
