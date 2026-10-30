# ============================================================
# TP2b - Puerto Sur - Grupo 11 | CODIGO 14
# Replicas y diagnostico de warm-up (NO validacion definitiva)
# Ejecutar: python validacion_simulador.py
# Requiere: simulador_escenarios.py y dependencias de codigos 8-10
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt
from scipy import stats

import simulador_patio
import simulador_camiones
from simulador_escenarios import PuertoEscenario
from simulador_patio import cargar_datos

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Validacion'
SALIDA.mkdir(exist_ok=True)

REPLICAS = 8
DIAS_SIMULACION = 98
DIAS_WARMUP_CANDIDATOS = [0, 7, 14, 21, 28]
DIAS_WARMUP_PRINCIPAL = 14
DIAS_VACIADO = 21
SEMILLA_BASE = 20261029
BERMAS = 3
STS = 3


def generar_llegadas_por_bloques(historico, rng, dias=DIAS_SIMULACION):
    """Remuestrear SEMANAS historicas completas con reemplazo.
    Mantiene patrones intrase manales y carga real por escala.
    Los bloques concatenados NO reproducen calendario estacional completo.
    """
    h = historico.copy()
    inicio_datos = h['fecha_hora_arribo'].min().normalize()
    h['semana'] = ((h['fecha_hora_arribo'] - inicio_datos).dt.total_seconds() // (7 * 86400)).astype(int)
    bloques = []
    for semana, grupo in h.groupby('semana'):
        # Solo semanas completas, no truncadas por fin de datos
        if inicio_datos + pd.Timedelta(days=7 * (int(semana) + 1)) <= h['fecha_hora_arribo'].max().normalize():
            g = grupo.copy()
            inicio = inicio_datos + pd.Timedelta(days=7 * int(semana))
            g['offset_h'] = (g['fecha_hora_arribo'] - inicio).dt.total_seconds() / 3600
            bloques.append(g)
    if not bloques:
        raise RuntimeError('No se encontraron semanas historicas completas')

    filas = []
    n_semanas = int(np.ceil(dias / 7))
    origen = pd.Timestamp('2025-01-01 00:00')
    for i in range(n_semanas):
        bloque = bloques[int(rng.integers(0, len(bloques)))].copy()
        bloque['hora_sim'] = i * 168.0 + bloque['offset_h']
        bloque = bloque[bloque['hora_sim'] < dias * 24].copy()
        bloque['fecha_hora_arribo'] = origen + pd.to_timedelta(bloque['hora_sim'], unit='h')
        bloque['id_escala'] = [f'R{i:03d}-{j:05d}' for j in range(len(bloque))]
        filas.append(bloque)

    datos = pd.concat(filas, ignore_index=True)
    return datos.sort_values('hora_sim').reset_index(drop=True), origen


def fuente(env, puerto, datos):
    for fila in datos.to_dict('records'):
        yield env.timeout(max(0., float(fila['hora_sim']) - env.now))
        env.process(puerto.atender(fila))


def ejecutar_replica(numero, historico):
    semilla = SEMILLA_BASE + numero
    rng = np.random.default_rng(semilla)
    datos, origen = generar_llegadas_por_bloques(historico, rng)

    # Resembrar fuentes de aleatoriedad que usan los modelos anteriores.
    simulador_patio.RNG.bit_generator.state = np.random.default_rng(semilla + 10000).bit_generator.state
    simulador_camiones.SEMILLA_CAMIONES = semilla + 20000
    simulador_camiones.POLITICA = 'actual'

    env = simpy.Environment()
    puerto = PuertoEscenario(env, origen, BERMAS, STS)
    env.process(fuente(env, puerto, datos))
    env.run(until=DIAS_SIMULACION * 24 + DIAS_VACIADO * 24)

    r = pd.DataFrame(puerto.registros)
    if r.empty:
        raise RuntimeError(f'Replica {numero}: sin buques completados')

    # Filtrar por instante de ARRIBO, no por finalizacion.
    # Las escalas pendientes son reportadas para evitar sesgo por censura.
    r['arribo_h'] = (pd.to_datetime(r['fecha_hora_arribo']) - origen).dt.total_seconds() / 3600
    resultados = []
    for warmup in DIAS_WARMUP_CANDIDATOS:
        esperado = datos[datos['hora_sim'] >= 24 * warmup]
        terminados = r[r['arribo_h'] >= 24 * warmup]
        pendientes = len(esperado) - len(terminados)
        resultados.append({
            'replica': numero,
            'warmup_dias': warmup,
            'ingresadas_poswarmup': len(esperado),
            'finalizadas_poswarmup': len(terminados),
            'pendientes_poswarmup': pendientes,
            'estadia_media_h': terminados['estadia_total_h'].mean() if len(terminados) else np.nan,
            'espera_media_h': terminados['espera_rada_h'].mean() if len(terminados) else np.nan,
            'cumplimiento_16_pct': 100 * (terminados['estadia_total_h'] <= 16).mean() if len(terminados) else np.nan,
        })

    r['replica'] = numero
    r.to_csv(SALIDA / f'replica_{numero:02d}_buques.csv', index=False)
    return resultados


def intervalo_t(valores):
    v = pd.Series(valores).dropna().to_numpy(dtype=float)
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    media = float(v.mean())
    semi = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return media, media - semi, media + semi


def main():
    historico, _ = cargar_datos()
    todos = []
    for i in range(1, REPLICAS + 1):
        print(f'Ejecutando replica {i}/{REPLICAS}...', flush=True)
        todos.extend(ejecutar_replica(i, historico))

    tabla = pd.DataFrame(todos)
    tabla.to_csv(SALIDA / 'resultados_replicas_y_warmup.csv', index=False)
    resumen = []
    for w, grupo in tabla.groupby('warmup_dias'):
        fila = {'warmup_dias': int(w), 'replicas': len(grupo),
                'pendientes_total': int(grupo['pendientes_poswarmup'].sum())}
        for col in ('estadia_media_h', 'espera_media_h', 'cumplimiento_16_pct'):
            media, bajo, alto = intervalo_t(grupo[col])
            fila[f'{col}_media'] = media
            fila[f'{col}_ic95_inf'] = bajo
            fila[f'{col}_ic95_sup'] = alto
        resumen.append(fila)
    res = pd.DataFrame(resumen).sort_values('warmup_dias')
    res.to_csv(SALIDA / 'resumen_warmup_ic95.csv', index=False)

    plt.figure(figsize=(9, 5))
    plt.errorbar(res['warmup_dias'], res['estadia_media_h_media'],
                 yerr=[res['estadia_media_h_media'] - res['estadia_media_h_ic95_inf'],
                       res['estadia_media_h_ic95_sup'] - res['estadia_media_h_media']],
                 marker='o', capsize=4)
    plt.axhline(16, linestyle='--', color='black', label='Umbral de 16 h')
    plt.xlabel('Dias descartados como warm-up')
    plt.ylabel('Estadia media (h), IC 95% entre replicas')
    plt.title('Diagnostico exploratorio del warm-up')
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(SALIDA / '01_warmup_estadia.png', dpi=180)
    plt.close()

    elegido = res[res['warmup_dias'] == DIAS_WARMUP_PRINCIPAL].iloc[0]
    print('\n=== CODIGO 14: REPLICAS Y WARM-UP ===')
    print(res[['warmup_dias', 'estadia_media_h_media', 'estadia_media_h_ic95_inf',
               'estadia_media_h_ic95_sup', 'pendientes_total']].to_string(index=False))
    print(f'\nWarm-up de referencia: {DIAS_WARMUP_PRINCIPAL} dias (NO validado automaticamente)')
    print(f'Estadia media: {elegido["estadia_media_h_media"]:.2f} h')
    print(f'IC 95% entre replicas: [{elegido["estadia_media_h_ic95_inf"]:.2f}, {elegido["estadia_media_h_ic95_sup"]:.2f}] h')
    print('Archivos guardados en:', SALIDA)
    print('ATENCION: remuestreo de semanas historicas + supuestos provisionales.')
    print('Esto evalua variabilidad y warm-up, pero NO valida el modelo contra datos reales de zarpada.')
    print('Si hay pendientes, los promedios de finalizadas estan censurados.')
    print('Falta verificar trazas internas de recursos y sensibilidad de supuestos.')


if __name__ == '__main__':
    main()
