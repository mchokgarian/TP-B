# TP2b Puerto Sur - Grupo 11 | Parte 16
# Interaccion camiones -> slots -> gruas -> estadia
# Requiere simulador_camiones.py, simulador_tractores.py, simulador_patio.py
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt

from simulador_camiones import Puerto as PuertoAnterior
from simulador_patio import cargar_datos, RNG as RNG_PUERTO
import simulador_patio

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Interaccion'
SALIDA.mkdir(parents=True, exist_ok=True)
SEMILLA = 51

# SUPUESTOS EXPLORATORIOS, NO DATOS DE LA CONSIGNA.
# La consigna solo proporciona un 20-30% de camiones impuntuales,
# permanencia media 72h, patio 3000 slots y congestion desde 75%.
ESCENARIOS = [
    dict(nombre='base', puertas=3, impuntuales=.25, atraso_extra_h=3, permanencia_h=72, ocupacion_inicial=1500, demanda=1.),
    dict(nombre='menos_puertas', puertas=1, impuntuales=.25, atraso_extra_h=3, permanencia_h=72, ocupacion_inicial=1500, demanda=1.),
    dict(nombre='mas_impuntuales', puertas=3, impuntuales=.50, atraso_extra_h=12, permanencia_h=72, ocupacion_inicial=1500, demanda=1.),
    dict(nombre='mas_permanencia', puertas=2, impuntuales=.50, atraso_extra_h=24, permanencia_h=96, ocupacion_inicial=2100, demanda=1.),
    dict(nombre='demanda_35_base', puertas=3, impuntuales=.25, atraso_extra_h=3, permanencia_h=72, ocupacion_inicial=1500, demanda=1.35),
    dict(nombre='demanda_35_demoras', puertas=1, impuntuales=.50, atraso_extra_h=24, permanencia_h=96, ocupacion_inicial=2100, demanda=1.35),
]


class Puerto(PuertoAnterior):
    def __init__(self, env, origen, config):
        # La ocupacion inicial usada por el modelo base se establece ANTES
        # de crear la instancia; se restaura tras la corrida.
        super().__init__(env, origen)
        self.config = config
        self.puertas = simpy.Resource(env, capacity=config['puertas'])
        self.rng_camiones = np.random.default_rng(510)
        self.horas_puertas = 0.
        self.bloqueos_patio_h = 0.
        self.camiones = []
        self.lotes_programados = 0

    def ingresar_contenedores(self, cantidad):
        inicio = self.env.now
        result = yield from super().ingresar_contenedores(cantidad)
        # El modelo de patio registra la espera de slots; esto es solo
        # un indicador agregado adicional del tiempo transcurrido.
        self.bloqueos_patio_h += max(0., self.env.now - inicio)
        return result

    def liberar_contenedores(self, cantidad):
        # El retiro no es automatico: comienza luego de permanencia
        # programada, puede atrasarse y espera una puerta libre.
        self.lotes_programados += 1
        c = self.config
        impuntual = self.rng_camiones.random() < c['impuntuales']
        atraso = self.rng_camiones.uniform(1, c['atraso_extra_h']) if impuntual else 0.
        yield self.env.timeout(c['permanencia_h'] + atraso)
        inicio_cola = self.env.now
        with self.puertas.request() as permiso:
            yield permiso
            espera = self.env.now - inicio_cola
            # Supuesto: una puerta tarda 10 min por lote de transporte.
            yield self.env.timeout(10 / 60)
            self.horas_puertas += 10 / 60
        self.actualizar_patio()
        self.ocupacion -= cantidad
        self.contenedores_retirados += cantidad
        yield self.slots_libres.put(cantidad)
        self.registrar_patio()
        self.camiones.append(dict(cantidad=cantidad, impuntual=impuntual,
                                  espera_cola_h=espera, atraso_h=atraso,
                                  hora_retiro=self.env.now))


def generador(env, puerto, datos):
    for i, fila in enumerate(datos.itertuples(index=False), start=1):
        # convertir a Series permite reutilizar las clases anteriores.
        f = pd.Series(fila._asdict())
        yield env.timeout(max(0., float(f['hora_sim']) - env.now))
        env.process(puerto.atender(f))
        if i % 2000 == 0:
            print(f'  Ingresaron {i}/{len(datos)}', flush=True)


def simular(config, datos_original, origen):
    # Se utiliza una ventana historica corta para que los escenarios
    # se puedan probar rapido sin perder las diferencias operativas.
    datos = datos_original.iloc[:1200].copy()
    datos['hora_sim'] = datos['hora_sim'] / config['demanda']

    RNG_PUERTO.bit_generator.state = np.random.default_rng(SEMILLA).bit_generator.state
    anterior = simulador_patio.OCUPACION_INICIAL
    simulador_patio.OCUPACION_INICIAL = config['ocupacion_inicial']
    try:
        env = simpy.Environment()
        puerto = Puerto(env, origen, config)
        env.process(generador(env, puerto, datos))
        # Cierre observacional: 14 dias despues de la ultima llegada.
        limite = float(datos['hora_sim'].max()) + 14*24
        env.run(until=limite)
        puerto.actualizar_patio()
    finally:
        simulador_patio.OCUPACION_INICIAL = anterior

    terminadas = pd.DataFrame(puerto.registros)
    pendientes = len(datos) - len(terminadas)
    if not terminadas.empty:
        terminadas.to_csv(SALIDA / (config['nombre'] + '_buques.csv'), index=False)
    df_camiones = pd.DataFrame(puerto.camiones)
    if not df_camiones.empty:
        df_camiones.to_csv(SALIDA / (config['nombre'] + '_camiones.csv'), index=False)

    return {
        'escenario': config['nombre'],
        'escalas': len(datos),
        'finalizadas': len(terminadas),
        'pendientes': pendientes,
        'estadia_media_finalizadas_h': terminadas['estadia_total_h'].mean() if len(terminadas) else np.nan,
        'espera_rada_finalizadas_h': terminadas['espera_rada_h'].mean() if len(terminadas) else np.nan,
        'ocupacion_media_pct': 100*puerto.area_ocupacion/(3000*env.now),
        'ocupacion_maxima_pct': 100*puerto.ocupacion_maxima/3000,
        'tiempo_sobre_75_pct': 100*puerto.tiempo_congestion/env.now,
        'bloqueos_patio_h_acum': puerto.bloqueos_patio_h,
        'espera_camion_h': df_camiones['espera_cola_h'].mean() if len(df_camiones) else np.nan,
        'lotes_camion_atendidos': len(df_camiones),
        'lotes_camion_pendientes': puerto.lotes_programados - len(df_camiones),
    }


def main():
    datos, origen = cargar_datos()
    filas = []
    for c in ESCENARIOS:
        print('Ejecutando:', c['nombre'], flush=True)
        filas.append(simular(c, datos, origen))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / 'comparacion_interaccion.csv', index=False)
    print('\n=== CODIGO 16: INTERACCION CAMIONES / PATIO / GRUAS ===')
    print(tabla.to_string(index=False, float_format=lambda n: f'{n:.3f}'))
    ax = tabla.plot(x='escenario', y='estadia_media_finalizadas_h', kind='bar', legend=False, figsize=(10,5))
    ax.set_ylabel('Estadia promedio de buques finalizados (h)')
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(SALIDA/'01_estadias.png', dpi=180)
    plt.close()
    ax = tabla.plot(x='escenario', y='ocupacion_maxima_pct', kind='bar', legend=False, figsize=(10,5))
    ax.axhline(75, ls='--', color='black')
    ax.set_ylabel('Ocupacion maxima del patio (%)')
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(SALIDA/'02_ocupacion.png', dpi=180)
    plt.close()
    print('\nResultados:', SALIDA)
    print('ATENCION: casos de sensibilidad; no son datos calibrados ni validacion empirica.')
    print('Las estadias de escenarios con pendientes estan censuradas.')
    print('El modelo solo genera bloqueo de gruas cuando se agotan slots;')
    print('superar el 75% se registra, pero la penalizacion de productividad')
    print('por congestion aun requiere una hipotesis cuantificada.')

if __name__ == '__main__':
    main()
