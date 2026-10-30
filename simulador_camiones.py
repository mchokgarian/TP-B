# TP2b - Puerto Sur - Grupo 11 | Codigo 10: camiones y ventanas
# Requiere simulador_tractores.py (version corregida, Codigo 9)
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt
from simulador_tractores import Puerto as PuertoTractores
from simulador_patio import cargar_datos, llegadas, CAPACIDAD_PATIO, BERMAS, STS_TOTALES

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Camiones'
SALIDA.mkdir(exist_ok=True)

# SUPUESTOS PROVISIONALES: la consigna solo especifica 20-30% impuntuales.
POLITICA = 'actual'  # 'actual', 'ventana_ampliada', 'penalizacion'
PUERTAS_CAMIONES = 3
TIEMPO_SERVICIO_H = 10 / 60
VENTANA_ACTUAL_H = 2.0
VENTANA_AMPLIADA_H = 4.0
PROB_IMPUNTUAL = 0.25
DESVIO_TARDE_H = (1.0, 4.0)
SEMILLA_CAMIONES = 510

class Puerto(PuertoTractores):
    def __init__(self, env, origen):
        super().__init__(env, origen)
        self.rng_camiones = np.random.default_rng(SEMILLA_CAMIONES)
        self.puertas = simpy.Resource(env, capacity=PUERTAS_CAMIONES)
        self.camiones = []
        self.horas_puertas = 0.0
        self.lotes_programados = 0

    def liberar_contenedores(self, cantidad):
        # Sustituye la liberacion automatica del Codigo 8 por una atencion
        # de camion. No altera la logica de descarga ni de tractores.
        self.lotes_programados += 1
        t_programado = self.env.now + 72.0  # permanencia media: supuesto fijo
        ventana = VENTANA_AMPLIADA_H if POLITICA == 'ventana_ampliada' else VENTANA_ACTUAL_H
        impuntual = self.rng_camiones.random() < PROB_IMPUNTUAL
        desviacion = self.rng_camiones.uniform(*DESVIO_TARDE_H) if impuntual else 0.0
        # Penalizacion: supuesto exploratorio de menor atraso (NO dato observado).
        if POLITICA == 'penalizacion':
            desviacion *= 0.5
        llegada_camion = t_programado + desviacion
        yield self.env.timeout(max(0.0, llegada_camion - self.env.now))
        inicio_cola = self.env.now
        with self.puertas.request() as permiso:
            yield permiso
            espera_cola = self.env.now - inicio_cola
            yield self.env.timeout(TIEMPO_SERVICIO_H)
            self.horas_puertas += TIEMPO_SERVICIO_H
        self.actualizar_patio()
        self.ocupacion -= cantidad
        self.contenedores_retirados += cantidad
        yield self.slots_libres.put(cantidad)
        self.registrar_patio()
        self.camiones.append({
            'cantidad': cantidad,
            'impuntual': int(impuntual),
            'fuera_ventana': int(desviacion > ventana),
            'atraso_h': desviacion,
            'espera_cola_h': espera_cola,
            'hora_retiro': self.env.now,
        })

def main():
    if POLITICA not in ('actual', 'ventana_ampliada', 'penalizacion'):
        raise ValueError('POLITICA no valida')
    datos, origen = cargar_datos()
    env = simpy.Environment()
    puerto = Puerto(env, origen)
    env.process(llegadas(env, puerto, datos))
    print('Iniciando simulacion con camiones...', flush=True)
    # Procesamos hasta el ultimo buque, pero no forzamos atender camiones
    # posteriores al cierre del horizonte como si hubieran llegado antes.
    limite = float(datos['hora_sim'].max()) + 1000.0
    env.run(until=limite)
    if len(puerto.registros) != len(datos):
        raise RuntimeError(f'Escalas atendidas: {len(puerto.registros)} de {len(datos)}. Revisar estabilidad.')
    puerto.actualizar_patio()
    barcos = pd.DataFrame(puerto.registros)
    camiones = pd.DataFrame(puerto.camiones)
    barcos.to_csv(SALIDA / 'resultados_por_buque.csv', index=False)
    camiones.to_csv(SALIDA / 'resultados_camiones.csv', index=False)
    horizonte = float(barcos['hora_salida_sim'].max())
    resumen = {
        'politica': POLITICA,
        'escalas_atendidas': len(barcos),
        'estadia_media_h': barcos['estadia_total_h'].mean(),
        'espera_media_rada_h': barcos['espera_rada_h'].mean(),
        'cumplimiento_16h_pct': 100 * (barcos['estadia_total_h'] <= 16).mean(),
        'utilizacion_bermas_pct': 100 * puerto.uso_bermas / (BERMAS * horizonte),
        'utilizacion_sts_pct': 100 * puerto.uso_gruas / (STS_TOTALES * horizonte),
        'ocupacion_media_patio_pct': 100 * puerto.area_ocupacion / (CAPACIDAD_PATIO * env.now),
        'ocupacion_maxima_patio_pct': 100 * puerto.ocupacion_maxima / CAPACIDAD_PATIO,
        'tiempo_sobre_75_pct': 100 * puerto.tiempo_congestion / env.now,
        'lotes_camiones_atendidos': len(camiones),
        'lotes_camiones_pendientes': puerto.lotes_programados - len(camiones),
        'impuntualidad_pct': 100 * camiones['impuntual'].mean(),
        'fuera_ventana_pct': 100 * camiones['fuera_ventana'].mean(),
        'espera_media_camion_h': camiones['espera_cola_h'].mean(),
        'utilizacion_puertas_pct': 100 * puerto.horas_puertas / (PUERTAS_CAMIONES * env.now),
    }
    pd.DataFrame([resumen]).to_csv(SALIDA / 'indicadores_resumen.csv', index=False)
    pd.DataFrame(puerto.historial_patio).to_csv(SALIDA / 'historial_patio.csv', index=False)
    plt.figure(figsize=(9, 4))
    plt.hist(camiones['espera_cola_h'] * 60, bins=35, edgecolor='black')
    plt.xlabel('Espera en puerta (minutos)')
    plt.ylabel('Lotes de camiones')
    plt.tight_layout()
    plt.savefig(SALIDA / '01_esperas_camiones.png', dpi=180)
    plt.close()
    h = pd.DataFrame(puerto.historial_patio)
    plt.figure(figsize=(10, 4))
    plt.plot(h['hora_sim'] / 24, h['ocupacion_pct'], lw=0.8)
    plt.axhline(75, linestyle='--', label='Umbral 75%')
    plt.xlabel('Dia de simulacion')
    plt.ylabel('Ocupacion de patio (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA / '02_ocupacion_patio.png', dpi=180)
    plt.close()
    print('\n=== CODIGO 10: CAMIONES Y VENTANAS ===')
    for k, v in resumen.items():
        print(f'{k}: {v:.3f}' if isinstance(v, (float, np.floating)) else f'{k}: {v}')
    print('\nResultados guardados en:', SALIDA)
    print('Supuestos de puertas, ventanas y atraso pendientes de sensibilidad/validacion.')

if __name__ == '__main__':
    main()
