# TP2b Puerto Sur - Grupo 11 | Parte 17: congestion de patio
# Requiere correccion_interaccion.py y los simuladores previos.
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt
import simulador_patio
from correccion_interaccion import Puerto as PuertoAnterior, generador

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Congestion'
SALIDA.mkdir(exist_ok=True)
SEMILLA = 51
UMBRAL = .75

# SOLO SUPUESTOS EXPLORATORIOS: la consigna no cuantifica
# la perdida de productividad cuando se supera 75%.
ESCENARIOS = [
    dict(nombre='base_sin_penalizacion', demanda=1., ocupacion_inicial=1500,
         puertas=3, impuntuales=.25, atraso_extra_h=3, permanencia_h=72, penalizacion=0.),
    dict(nombre='base_penalizacion_15', demanda=1., ocupacion_inicial=1500,
         puertas=3, impuntuales=.25, atraso_extra_h=3, permanencia_h=72, penalizacion=.15),
    dict(nombre='estres_sin_penalizacion', demanda=1., ocupacion_inicial=2100,
         puertas=2, impuntuales=.50, atraso_extra_h=24, permanencia_h=96, penalizacion=0.),
    dict(nombre='estres_penalizacion_15', demanda=1., ocupacion_inicial=2100,
         puertas=2, impuntuales=.50, atraso_extra_h=24, permanencia_h=96, penalizacion=.15),
    dict(nombre='estres_penalizacion_30', demanda=1., ocupacion_inicial=2100,
         puertas=2, impuntuales=.50, atraso_extra_h=24, permanencia_h=96, penalizacion=.30),
]

class Puerto(PuertoAnterior):
    def __init__(self, env, origen, config):
        super().__init__(env, origen, config)
        self.horas_sts_congestionadas = 0.
        self.horas_adicionales_penalizacion = 0.
        self.esperas_por_slots_h = 0.

    def operar(self, carga):
        # Conserva la logica de turnos y STS del Codigo 8.
        trabajo = sum(float(carga[k]) / tasa for k, tasa in simulador_patio.TASA.items())
        tiempo_efectivo = 0.
        horas_sts = 0.
        uso_dos = False
        while trabajo > 1e-8:
            yield from self.esperar_relevo()
            inicio, fin, noche = self.turno_actual()
            recurso_noche = None
            if noche:
                recurso_noche = self.recurso_nocturno(inicio)
                yield recurso_noche.get(1)
            yield self.gruas.get(1)
            n = 1
            if self.gruas.level >= 1:
                if not noche:
                    yield self.gruas.get(1)
                    n = 2
                elif recurso_noche.level >= 1:
                    yield recurso_noche.get(1)
                    yield self.gruas.get(1)
                    n = 2
            uso_dos = uso_dos or (n == 2)
            interferencia = .85 if n == 2 else 1.
            congestion = self.ocupacion / simulador_patio.CAPACIDAD_PATIO > UMBRAL
            penalizacion = self.config['penalizacion'] if congestion else 0.
            velocidad = n * interferencia * (1. - penalizacion)
            duracion = min(trabajo / velocidad, max(0., fin - self.calendario()))
            if duracion > 1e-9:
                yield self.env.timeout(duracion)
                trabajo -= duracion * velocidad
                tiempo_efectivo += duracion
                horas_sts += n * duracion
                self.uso_gruas += n * duracion
                if congestion:
                    self.horas_sts_congestionadas += n * duracion
                    # Equivalente respecto de operacion sin penalizacion.
                    self.horas_adicionales_penalizacion += duracion * penalizacion
            yield self.gruas.put(n)
            if noche:
                yield recurso_noche.put(n)
            if duracion <= 1e-9:
                yield self.env.timeout(1e-6)
        return tiempo_efectivo, horas_sts, uso_dos

    def ingresar_contenedores(self, cantidad):
        # Medir solo espera de slots, excluyendo desplazamiento de tractores.
        antes = len(self.esperas_patio)
        resultado = yield from super().ingresar_contenedores(cantidad)
        self.esperas_por_slots_h += sum(self.esperas_patio[antes:])
        return resultado


def simular(config, datos_orig, origen):
    datos = datos_orig.iloc[:1200].copy()
    datos['hora_sim'] /= config['demanda']
    simulador_patio.RNG.bit_generator.state = np.random.default_rng(SEMILLA).bit_generator.state
    inicial_anterior = simulador_patio.OCUPACION_INICIAL
    simulador_patio.OCUPACION_INICIAL = config['ocupacion_inicial']
    try:
        env = simpy.Environment()
        puerto = Puerto(env, origen, config)
        env.process(generador(env, puerto, datos))
        limite = float(datos['hora_sim'].max()) + 14 * 24
        env.run(until=limite)
        puerto.actualizar_patio()
    finally:
        simulador_patio.OCUPACION_INICIAL = inicial_anterior
    r = pd.DataFrame(puerto.registros)
    pendientes = len(datos) - len(r)
    salida = dict(
        escenario=config['nombre'], escalas=len(datos), finalizadas=len(r), pendientes=pendientes,
        estadia_media_finalizadas_h=r['estadia_total_h'].mean() if len(r) else np.nan,
        espera_rada_media_finalizadas_h=r['espera_rada_h'].mean() if len(r) else np.nan,
        ocupacion_maxima_pct=100*puerto.ocupacion_maxima/3000,
        tiempo_sobre_75_pct=100*puerto.tiempo_congestion/env.now,
        horas_sts_congestionadas=puerto.horas_sts_congestionadas,
        horas_adicionales_penalizacion=puerto.horas_adicionales_penalizacion,
        espera_slots_acumulada_h=puerto.esperas_por_slots_h,
        retiros_camion=len(puerto.camiones),
        lotes_camion_pendientes=puerto.lotes_programados-len(puerto.camiones),
    )
    if len(r):
        r.to_csv(SALIDA/(config['nombre']+'_buques.csv'),index=False)
    return salida


def main():
    datos, origen = simulador_patio.cargar_datos()
    resultados = []
    for config in ESCENARIOS:
        print('Ejecutando:',config['nombre'],flush=True)
        resultados.append(simular(config,datos,origen))
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(SALIDA/'comparacion_congestion.csv',index=False)
    print('\n=== CODIGO 17: CONGESTION DEL PATIO ===')
    print(tabla.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(tabla['escenario'],tabla['estadia_media_finalizadas_h'])
    ax.set_ylabel('Estadia media de buques finalizados (h)')
    ax.tick_params(axis='x',rotation=25)
    plt.tight_layout()
    fig.savefig(SALIDA/'01_estadias_congestion.png',dpi=180)
    plt.close(fig)
    print('\nResultados guardados en:',SALIDA)
    print('ATENCION: penalizaciones 0%, 15% y 30% son HIPOTESIS de sensibilidad, NO datos reales.')
    print('Los promedios con pendientes son censurados; el modelo todavia requiere validacion.')

if __name__=='__main__':
    main()
