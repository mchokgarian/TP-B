# TP2b Puerto Sur - Grupo 11 | CODIGO 15
# Verificacion operacional y sensibilidad (diagnostica, NO validacion empirica)
# Ejecutar: python verificacion_final.py
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt
import simulador_patio as patio
import simulador_tractores as tractores
import simulador_camiones as camiones

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Verificacion_Final'
SALIDA.mkdir(exist_ok=True)
SEMILLA = 51
N_ESCALAS_PRUEBA = 1200  # Submuestra para pruebas rapidas; NO extrapolar resultados


def escenario(nombre, datos, origen, n_tractores, viaje_min, n_puertas, impuntualidad):
    # Cambios temporales de parametros, sin modificar archivos originales.
    tractores.TRACTORES_DIA = n_tractores
    tractores.TIEMPO_VIAJE_MIN = viaje_min
    camiones.PUERTAS_CAMIONES = n_puertas
    camiones.PROB_IMPUNTUAL = impuntualidad
    camiones.POLITICA = 'actual'
    patio.RNG.bit_generator.state = np.random.default_rng(SEMILLA).bit_generator.state

    class PuertoAuditado(camiones.Puerto):
        def __init__(self, env, origen):
            super().__init__(env, origen)
            self.infracciones = []
            self.muestras = 0

        def auditar(self):
            self.muestras += 1
            t = self.env.now
            if not (0 <= self.ocupacion <= patio.CAPACIDAD_PATIO):
                self.infracciones.append((t, 'patio fuera de capacidad', self.ocupacion))
            if not (0 <= self.gruas.level <= self.gruas.capacity):
                self.infracciones.append((t, 'STS tokens fuera de capacidad', self.gruas.level))
            if not (0 <= self.bermas.count <= self.bermas.capacity):
                self.infracciones.append((t, 'bermas fuera de capacidad', self.bermas.count))
            if not (0 <= self.tractores.count <= self.tractores.capacity):
                self.infracciones.append((t, 'tractores fuera de capacidad', self.tractores.count))
            if not (0 <= self.puertas.count <= self.puertas.capacity):
                self.infracciones.append((t, 'puertas fuera de capacidad', self.puertas.count))

        def registrar_patio(self):
            super().registrar_patio()
            self.auditar()

    env = simpy.Environment()
    puerto = PuertoAuditado(env, origen)
    env.process(patio.llegadas(env, puerto, datos))
    print('Ejecutando:', nombre, flush=True)
    fin = float(datos['hora_sim'].max()) + 1000.0
    env.run(until=fin)
    puerto.auditar()
    barcos = pd.DataFrame(puerto.registros)
    pendientes = len(datos) - len(barcos)
    puerto.actualizar_patio()

    errores = puerto.infracciones[:]
    if not barcos.empty:
        for columna in ['estadia_total_h', 'espera_rada_h']:
            if not np.isfinite(barcos[columna]).all() or (barcos[columna] < -1e-8).any():
                errores.append((env.now, 'tiempo invalido', columna))
        if (barcos['estadia_total_h'] + 1e-7 < barcos['espera_rada_h']).any():
            errores.append((env.now, 'estadias menores que espera', ''))
        if barcos['id_escala'].duplicated().any():
            errores.append((env.now, 'IDs duplicados', ''))

    resumen = {
        'escenario': nombre,
        'escalas': len(datos),
        'finalizadas': len(barcos),
        'pendientes': pendientes,
        'estadia_media_finalizadas_h': barcos['estadia_total_h'].mean() if len(barcos) else np.nan,
        'espera_rada_media_finalizadas_h': barcos['espera_rada_h'].mean() if len(barcos) else np.nan,
        'ocupacion_maxima_patio_pct': 100 * puerto.ocupacion_maxima / patio.CAPACIDAD_PATIO,
        'espera_tractor_media_h': np.mean(puerto.esperas_tractores) if puerto.esperas_tractores else np.nan,
        'espera_camion_media_h': np.mean([c['espera_cola_h'] for c in puerto.camiones]) if puerto.camiones else np.nan,
        'viajes_tractores': puerto.viajes_realizados,
        'camiones_atendidos': len(puerto.camiones),
        'auditorias': puerto.muestras,
        'infracciones_detectadas': len(errores),
    }
    if errores:
        pd.DataFrame(errores, columns=['hora_sim','tipo','valor']).to_csv(
            SALIDA / f'errores_{nombre}.csv', index=False
        )
    return resumen


def main():
    originales = {
        'tractores_dia': tractores.TRACTORES_DIA,
        'tiempo_viaje': tractores.TIEMPO_VIAJE_MIN,
        'puertas': camiones.PUERTAS_CAMIONES,
        'prob_impuntual': camiones.PROB_IMPUNTUAL,
        'politica': camiones.POLITICA,
    }
    datos, origen = patio.cargar_datos()
    datos = datos.iloc[:N_ESCALAS_PRUEBA].copy()
    configuraciones = [
        ('base', 4, 15, 3, 0.25),
        ('menos_tractores', 2, 15, 3, 0.25),
        ('viajes_mas_largos', 4, 25, 3, 0.25),
        ('menos_puertas', 4, 15, 1, 0.25),
        ('mas_impuntuales', 4, 15, 3, 0.30),
    ]
    try:
        resultados = [escenario(n, datos, origen, nt, tv, np_, pi)
                      for n, nt, tv, np_, pi in configuraciones]
    finally:
        tractores.TRACTORES_DIA = originales['tractores_dia']
        tractores.TIEMPO_VIAJE_MIN = originales['tiempo_viaje']
        camiones.PUERTAS_CAMIONES = originales['puertas']
        camiones.PROB_IMPUNTUAL = originales['prob_impuntual']
        camiones.POLITICA = originales['politica']

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(SALIDA / 'sensibilidad_1200_escalas.csv', index=False)
    print('\n=== CODIGO 15: VERIFICACION Y SENSIBILIDAD ===')
    print(tabla.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nTotal de infracciones detectadas:', tabla['infracciones_detectadas'].sum())
    print('Total de escenarios con pendientes:', int((tabla['pendientes'] > 0).sum()))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(tabla['escenario'], tabla['estadia_media_finalizadas_h'])
    ax.axhline(16, ls='--', color='black', label='Umbral 16 h')
    ax.set_ylabel('Estadia media de escalas finalizadas (h)')
    ax.set_title('Sensibilidad exploratoria - primeras 1200 escalas')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / '01_sensibilidad_estadia.png', dpi=180)
    plt.close(fig)

    (SALIDA / 'limitaciones.txt').write_text(
        'VERIFICACION PARCIAL: los controles de recursos observan estados y rangos, no todas las transiciones.\n'
        'PENDIENTE: trazas por grua individual, turnos nocturnos, relevo y standby.\n'
        'PENDIENTE: demostrar conservacion de slots en cada evento, incluidos contenedores iniciales.\n'
        'PENDIENTE: justificar capacidad/viaje de tractores, puertas y ventanas con fuentes o sensibilidad.\n'
        'PENDIENTE: tratar el hueco historico de 636 horas en el generador de llegadas.\n'
        'PENDIENTE: validar contra datos externos; no hay zarpadas historicas en el CSV.\n'
        'RESULTADOS: primeros 1200 registros, NO extrapolables a 3 anos ni a picos estacionales.\n'
        'AUSENCIA DE INFRACCIONES NO SIGNIFICA QUE EL SIMULADOR ESTE VALIDADO.\n',
        encoding='utf-8'
    )
    print('\nArchivos guardados en:', SALIDA)
    print('ATENCION: diagnostico parcial, NO validacion del modelo completo.')


if __name__ == '__main__':
    main()
