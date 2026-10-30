# TP2b Puerto Sur - Grupo 11 | PARTE 20: validacion estadistica interna
# Ejecutar junto a simulador_final.py y PuertoSur_escalas_G11.csv
# Las cifras de sensibilidad NO son parametros calibrados.
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import simpy
from scipy.stats import t as student_t
import simulador_final as modelo

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Validacion_Final'
SALIDA.mkdir(parents=True, exist_ok=True)
REPLICAS = 8
SEMANAS_POR_REPLICA = 12
CALENTAMIENTO_DIAS = 7
DIAS_CIERRE = 14
SEMILLA_MUESTREO = 202620

# Contrastes pareados sobre exactamente la misma muestra de demanda por replica.
# El escenario 'sensibilidad_tractores' utiliza un supuesto alternativo, no medido.
CONFIGS = {c['nombre']: deepcopy(c) for c in modelo.ESCENARIOS}
PRUEBAS = [
    ('base', CONFIGS['base']),
    ('cuarta_sts', CONFIGS['cuarta_sts']),
    ('cuarta_berma', CONFIGS['cuarta_berma']),
    ('ambas', CONFIGS['ambas']),
    ('congestion_15', CONFIGS['congestion_15']),
    ('sensibilidad_tractores', CONFIGS['base']),
]


def muestrear_semanas(df, rng, n_semanas=SEMANAS_POR_REPLICA):
    """Bootstrap de bloques de 7 dias; conserva orden y cargas dentro de cada semana."""
    datos = df.copy()
    datos['semana'] = np.floor(datos['hora_sim'] / 168).astype(int)
    bloques = []
    # Evitar la ultima semana incompleta.
    sem_max = int(datos['semana'].max())
    for k, bloque in datos.groupby('semana', sort=True):
        if k >= sem_max or len(bloque) == 0:
            continue
        bloques.append(bloque.copy())
    if not bloques:
        raise ValueError('No se pudieron construir semanas historicas completas.')
    muestra = []
    for j in range(n_semanas):
        elegido = bloques[int(rng.integers(0, len(bloques)))].copy()
        semana_orig = int(elegido['semana'].iloc[0])
        elegido['hora_sim'] = elegido['hora_sim'] - 168 * semana_orig + 168 * j
        elegido['id_escala'] = [f'R{j:02d}-{i:05d}' for i in range(len(elegido))]
        muestra.append(elegido)
    salida = pd.concat(muestra, ignore_index=True)
    salida = salida.sort_values('hora_sim', kind='stable').reset_index(drop=True)
    # Fecha sintetica para presentacion, no observacion historica real.
    inicio = pd.Timestamp('2025-01-01')
    salida['fecha_hora_arribo'] = inicio + pd.to_timedelta(salida['hora_sim'], unit='h')
    return salida.drop(columns=['semana']), inicio


def ejecutar_replica(nombre, config, datos, origen, semilla, tractores_lentos=False):
    """No llama a modelo.ejecutar: evita su semilla fija y sus CSV por corrida."""
    modelo.BERMAS = int(config['bermas'])
    modelo.STS_TOTALES = int(config['sts'])
    modelo.OCUPACION_INICIAL = int(config['ocupacion_inicial'])
    modelo.RNG.bit_generator.state = np.random.default_rng(semilla).bit_generator.state
    anterior_viaje = modelo.TIEMPO_VIAJE_MIN
    if tractores_lentos:
        modelo.TIEMPO_VIAJE_MIN = 20  # HIPOTESIS: 20 minutos vs 15 de la base.
    try:
        entradas = datos.copy()
        entradas['hora_sim'] /= float(config['demanda'])
        env = simpy.Environment()
        puerto = modelo.PuertoFinal(env, origen, config)
        # Simulador integrado fija la semilla de camiones dentro de __init__.
        # La cambiamos por replica para considerar su incertidumbre.
        puerto.rng_camiones = np.random.default_rng(semilla + 1_000_000)
        env.process(modelo.llegadas(env, puerto, entradas))
        limite = float(entradas['hora_sim'].max()) + 24 * DIAS_CIERRE
        env.run(until=limite)
        puerto.actualizar_patio()
        r = pd.DataFrame(puerto.registros)
        # Calentamiento por fecha de llegada, no por fecha de finalizacion.
        if len(r):
            r['hora_llegada_sim'] = r['hora_salida_sim'] - r['estadia_total_h']
            observados = r[r['hora_llegada_sim'] >= CALENTAMIENTO_DIAS * 24]
        else:
            observados = r
        n_llegadas_eval = int((entradas['hora_sim'] >= CALENTAMIENTO_DIAS * 24).sum())
        n_fin = len(observados)
        pendientes_eval = n_llegadas_eval - n_fin
        return {
            'escenario': nombre,
            'ingresadas': len(entradas), 'finalizadas': len(r),
            'pendientes': len(entradas) - len(r),
            'ingresadas_post_warmup': n_llegadas_eval,
            'finalizadas_post_warmup': n_fin,
            'pendientes_post_warmup': pendientes_eval,
            'estadia_media_h': float(observados['estadia_total_h'].mean()) if n_fin else np.nan,
            'espera_rada_media_h': float(observados['espera_rada_h'].mean()) if n_fin else np.nan,
            'cumple_16h_pct': 100 * float((observados['estadia_total_h'] <= 16).mean()) if n_fin else np.nan,
            'ocupacion_maxima_patio_pct': 100 * puerto.ocupacion_maxima / modelo.CAPACIDAD_PATIO,
            'tiempo_sobre_75_pct': 100 * puerto.tiempo_congestion / env.now,
            'utilizacion_bermas_pct': 100 * puerto.uso_bermas / (int(config['bermas']) * env.now),
            'utilizacion_sts_pct': 100 * puerto.uso_gruas / (int(config['sts']) * env.now),
        }
    finally:
        modelo.TIEMPO_VIAJE_MIN = anterior_viaje


def resumen_intervalo(valores):
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (len(v), np.nan, np.nan, np.nan)
    media = float(v.mean())
    sem = float(v.std(ddof=1) / math.sqrt(len(v)))
    margen = float(student_t.ppf(.975, df=len(v)-1) * sem)
    return (len(v), media, media-margen, media+margen)


def main():
    originales, _ = modelo.cargar_datos()
    maestro = np.random.default_rng(SEMILLA_MUESTREO)
    filas = []
    for rep in range(1, REPLICAS+1):
        bloque, origen = muestrear_semanas(originales, maestro)
        semilla = 31000 + rep * 100
        print(f'Replica {rep}/{REPLICAS}: {len(bloque)} buques, {SEMANAS_POR_REPLICA} semanas', flush=True)
        for nombre, config in PRUEBAS:
            print('  Ejecutando:', nombre, flush=True)
            fila = ejecutar_replica(nombre, config, bloque, origen, semilla,
                                    tractores_lentos=(nombre == 'sensibilidad_tractores'))
            fila['replica'] = rep
            filas.append(fila)
            print('   Finalizadas:', fila['finalizadas'], 'Pendientes:', fila['pendientes'], flush=True)

    detalle = pd.DataFrame(filas)
    detalle.to_csv(SALIDA/'replicas_detalle.csv', index=False)
    metricas = ['estadia_media_h', 'espera_rada_media_h', 'cumple_16h_pct',
                'ocupacion_maxima_patio_pct', 'utilizacion_sts_pct']
    resumen = []
    for (escenario, metrica), grupo in (
        ( (e, m), detalle.loc[detalle['escenario'].eq(e), m] )
        for e in detalle['escenario'].unique() for m in metricas
    ):
        n, media, li, ls = resumen_intervalo(grupo)
        resumen.append(dict(escenario=escenario, metrica=metrica,
                            replicas=n, media=media, ic95_inferior=li, ic95_superior=ls,
                            alguna_replica_pendiente=bool((detalle.loc[detalle['escenario'].eq(escenario), 'pendientes_post_warmup']>0).any())))
    tabla = pd.DataFrame(resumen)
    tabla.to_csv(SALIDA/'intervalos_confianza_95.csv', index=False)

    # Diferencias pareadas: + implica que la alternativa aumenta la estadia respecto a la base.
    pivote = detalle.pivot(index='replica', columns='escenario', values='estadia_media_h')
    diferencias = []
    for otro in [e for e in pivote.columns if e != 'base']:
        n, media, li, ls = resumen_intervalo(pivote[otro]-pivote['base'])
        diferencias.append(dict(comparacion=f'{otro} - base', replicas=n,
                                diferencia_media_h=media, ic95_inferior_h=li,
                                ic95_superior_h=ls))
    pd.DataFrame(diferencias).to_csv(SALIDA/'comparaciones_pareadas.csv', index=False)

    print('\n=== PARTE 20: VALIDACION ESTADISTICA INTERNA ===')
    mostrar = tabla[tabla['metrica'].eq('estadia_media_h')]
    print(mostrar[['escenario','media','ic95_inferior','ic95_superior','alguna_replica_pendiente']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\n=== DIFERENCIAS PAREADAS DE ESTADIA RESPECTO A BASE ===')
    print(pd.DataFrame(diferencias).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nCarpeta:', SALIDA)
    print('ADVERTENCIA: intervalos entre replicas bootstrap, NO validacion empirica.')
    print('Las semanas historicas remuestreadas no son replicas fisicamente independientes.')
    print('IC de estadia con pendientes: descriptivo de FINALIZADAS, CENSURADO.')
    print('La ocupacion inicial, los tractores y las penalizaciones siguen siendo supuestos.')
    print('Los seis controles pendientes de Parte 19 NO quedan resueltos por esta prueba.')

if __name__ == '__main__':
    main()
