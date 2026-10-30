# TP2b - Puerto Sur - Grupo 11
# Codigo 4: estimacion a priori mediante teoria de filas (M/M/3)
# IMPORTANTE: aproximacion, no reemplaza al simulador de eventos discretos.

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CSV = BASE / 'PuertoSur_escalas_G11.csv'
SALIDA = BASE / 'Resultados_Modelo_Filas'
SALIDA.mkdir(exist_ok=True)

# DATOS DE LA CONSIGNA
BERMAS = 3
GRUAS = 3
RENDIMIENTOS = {'mov_dry': 28.0, 'mov_reefer': 21.0, 'mov_oog': 12.0}  # mov/h por grua
MANIOBRA_TOTAL_MIN = 1.0  # horas, atraque + desatraque
MANIOBRA_TOTAL_MAX = 2.0
# SUPUESTO PARA ESTE MODELO A PRIORI: valor medio del intervalo, 1.5 h.
MANIOBRA_MEDIA = (MANIOBRA_TOTAL_MIN + MANIOBRA_TOTAL_MAX) / 2


def mmc(lam, tiempo_servicio, c):
    """Erlang C para una cola M/M/c con capacidad ilimitada."""
    mu = 1 / tiempo_servicio
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return {'rho': rho, 'P_espera': None, 'Wq_h': None,
                'W_h': None, 'Lq': None, 'L': None, 'estable': False}
    p0 = 1 / (sum(a**n / math.factorial(n) for n in range(c))
              + a**c / (math.factorial(c) * (1-rho)))
    p_espera = (a**c / (math.factorial(c) * (1-rho))) * p0
    wq = p_espera / (c*mu - lam)
    w = wq + tiempo_servicio
    return {'rho': rho, 'P_espera': p_espera, 'Wq_h': wq,
            'W_h': w, 'Lq': lam*wq, 'L': lam*w, 'estable': True}


# 1. HISTORICO DE ESCALAS
if not CSV.exists():
    raise FileNotFoundError(f'No encuentro {CSV}. Coloca el CSV junto a este .py')

df = pd.read_csv(CSV, parse_dates=['fecha_hora_arribo'])
cols = ['mov_dry', 'mov_reefer', 'mov_oog']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='raise')
if df[cols].isna().any().any() or (df[cols] < 0).any().any():
    raise ValueError('Hay movimientos negativos o valores faltantes en el CSV')

inicio = df['fecha_hora_arribo'].min().normalize()
fin = df['fecha_hora_arribo'].max().normalize()
dias_observados = (fin-inicio).days + 1
llegadas_dia = len(df)/dias_observados
lambda_h = llegadas_dia/24

# 2. TIEMPO DE SERVICIO IDEAL DE UNA STS POR BUQUE
# Cada movimiento requiere 1/productividad horas de grua.
# Se supone 1 STS por buque, sin interrupciones nocturnas,
# colas de gruas, tractores ni congestion de patio.
df['h_sts_ideal'] = sum(df[col] / velocidad for col, velocidad in RENDIMIENTOS.items())
df['h_servicio_aprox'] = MANIOBRA_MEDIA + df['h_sts_ideal']
servicio_media = df['h_servicio_aprox'].mean()

# 3. M/M/3: tres bermas con una grua dedicada por buque
base = mmc(lambda_h, servicio_media, BERMAS)

print('\n=== CODIGO 4: ESTIMACION A PRIORI (M/M/3) ===')
print(f'Escalas historicas: {len(df)}')
print(f'Dias observados: {dias_observados}')
print(f'Tasa media: {llegadas_dia:.3f} buques/dia = {lambda_h:.4f} buques/h')
print(f'Tiempo medio de operacion ideal con 1 STS: {df["h_sts_ideal"].mean():.3f} h')
print(f'Tiempo medio de maniobras (SUPUESTO): {MANIOBRA_MEDIA:.3f} h')
print(f'Tiempo medio de servicio aproximado: {servicio_media:.3f} h')
print(f'Utilizacion estimada (rho): {base["rho"]:.3%}')
if base['estable']:
    print(f'Probabilidad de esperar: {base["P_espera"]:.3%}')
    print(f'Espera media en rada Wq: {base["Wq_h"]:.3f} h')
    print(f'Estadia media total W: {base["W_h"]:.3f} h')
    print(f'Buques promedio en cola Lq: {base["Lq"]:.3f}')
    print(f'Buques promedio en sistema L: {base["L"]:.3f}')
else:
    print('M/M/3 inestable: lambda >= 3 * mu; no hay espera estacionaria finita.')

# 4. ESCENARIOS DE DEMANDA DE LA CONSIGNA (solo sensibilidad analitica)
filas = []
for nombre, factor in [('Actual', 1.0), ('Proyecto +35%', 1.35), ('Proyecto +100%', 2.0)]:
    lam = lambda_h * factor
    resultados = mmc(lam, servicio_media, BERMAS)
    filas.append({'escenario': nombre, 'factor_demanda': factor,
                  'lambda_buques_h': lam, 'servidores': BERMAS,
                  'servicio_medio_h': servicio_media, **resultados})
resumen = pd.DataFrame(filas)
resumen.to_csv(SALIDA / 'estimaciones_mmc.csv', index=False)

print('\n=== ESCENARIOS DE DEMANDA ===')
print(resumen[['escenario', 'rho', 'estable', 'Wq_h', 'W_h', 'L']].to_string(index=False))

# 5. GRAFICOS
plt.figure(figsize=(8, 5))
plt.bar(resumen['escenario'], resumen['rho'] * 100)
plt.axhline(100, linestyle='--', color='red', label='Limite de estabilidad (100%)')
plt.ylabel('Utilizacion teorica (%)')
plt.title('M/M/3 - Sensibilidad a la demanda')
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / '01_utilizacion_por_escenario.png', dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
validos = resumen[resumen['estable']]
plt.bar(validos['escenario'], validos['W_h'])
plt.axhline(16, linestyle='--', color='red', label='Umbral de 16 h')
plt.ylabel('Estadia teorica media (h)')
plt.title('M/M/3 - Estadia estimada (escenarios estables)')
plt.legend()
plt.tight_layout()
plt.savefig(SALIDA / '02_estadia_escenarios_estables.png', dpi=200)
plt.close()

print(f'\nArchivos guardados en: {SALIDA}')
print('ADVERTENCIA: M/M/3 asume arribos Poisson homogeneos y servicios exponenciales,')
print('1 STS por buque y sin restricciones de turnos/patio/tractores.')
print('La capacidad adicional de una 4a STS no puede representarse correctamente')
print('como una cuarta berma. Se evaluara mediante SimPy.')
print('Un escenario M/M/3 inestable NO significa necesariamente que el puerto real lo sea.')
