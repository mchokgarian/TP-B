# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 5: PRIMER SIMULADOR DE EVENTOS DISCRETOS (SimPy)
# Version inicial: bermas, STS, atraque, operacion y desatraque.
# NO incluye aun turnos, patio, tractores, camiones o cruceros.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import simpy

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / 'PuertoSur_escalas_G11.csv'
SALIDA = BASE / 'Resultados_Simulador_Base'
SALIDA.mkdir(exist_ok=True)

# Parametros del escenario actual
BERMAS = 3
GRUAS_STS = 3
MAX_STS_POR_BUQUE = 2
RENDIMIENTOS = {'mov_dry': 28.0, 'mov_reefer': 21.0, 'mov_oog': 12.0}  # mov/h por STS
FACTOR_DOS_STS = 0.85  # interferencia conjunta cuando trabajan dos STS
SEMILLA = 51

# SUPUESTO PROVISIONAL: maniobra conjunta Uniforme(1,2) horas.
# Se reparte en dos mitades (atraque/desatraque), porque la consigna
# no especifica una distribucion ni la particion entre maniobras.
RNG = np.random.default_rng(SEMILLA)


def cargar_escalas():
    datos = pd.read_csv(ARCHIVO, parse_dates=['fecha_hora_arribo'])
    columnas = ['id_escala', 'fecha_hora_arribo', 'servicio',
                'mov_dry', 'mov_reefer', 'mov_oog']
    datos = datos[columnas].sort_values('fecha_hora_arribo').reset_index(drop=True)
    if datos[columnas].isna().any().any():
        raise ValueError('Hay valores faltantes en el CSV de escalas.')
    if (datos[['mov_dry', 'mov_reefer', 'mov_oog']] < 0).any().any():
        raise ValueError('Se encontraron movimientos negativos.')
    if datos['id_escala'].duplicated().any():
        raise ValueError('Se encontraron IDs de escala duplicados.')
    # Se reproducen exactamente las fechas historicas, incluido el hueco
    # excepcional de marzo/abril de 2024. No se ajusta distribucion aun.
    inicio = datos['fecha_hora_arribo'].min()
    datos['hora_sim'] = (datos['fecha_hora_arribo'] - inicio).dt.total_seconds() / 3600.0
    return datos


class Puerto:
    def __init__(self, env):
        self.env = env
        self.bermas = simpy.Resource(env, capacity=BERMAS)
        self.sts = simpy.Container(env, capacity=GRUAS_STS, init=GRUAS_STS)
        self.resultados = []
        self.max_cola_rada = 0
        self.berma_horas = 0.0
        self.sts_horas = 0.0

    def atender_buque(self, fila):
        llegada = self.env.now
        self.max_cola_rada = max(self.max_cola_rada, len(self.bermas.queue))
        with self.bermas.request() as permiso:
            yield permiso
            espera_rada = self.env.now - llegada
            inicio_berma = self.env.now

            maniobras = RNG.uniform(1.0, 2.0)
            yield self.env.timeout(maniobras / 2.0)  # Atraque

            # Espera por la primera STS, con la berma ya ocupada.
            yield self.sts.get(1)
            n_sts = 1
            # Si existe otra STS libre al comenzar, se incorpora.
            # No se espera por una segunda STS: es una politica inicial.
            if self.sts.level >= 1:
                yield self.sts.get(1)
                n_sts = 2

            trabajo_grua_h = sum(
                float(fila[col]) / rendimiento
                for col, rendimiento in RENDIMIENTOS.items()
            )
            productividad_equivalente = n_sts * (FACTOR_DOS_STS if n_sts == 2 else 1.0)
            duracion_operacion = trabajo_grua_h / productividad_equivalente
            yield self.env.timeout(duracion_operacion)
            self.sts_horas += n_sts * duracion_operacion
            yield self.sts.put(n_sts)

            yield self.env.timeout(maniobras / 2.0)  # Desatraque
            salida = self.env.now
            self.berma_horas += salida - inicio_berma

        self.resultados.append({
            'id_escala': fila['id_escala'],
            'servicio': fila['servicio'],
            'fecha_hora_arribo': fila['fecha_hora_arribo'],
            'mov_total': int(fila['mov_dry'] + fila['mov_reefer'] + fila['mov_oog']),
            'sts_asignadas': n_sts,
            'espera_rada_h': espera_rada,
            'operacion_h': duracion_operacion,
            'maniobras_h': maniobras,
            'estadia_total_h': salida - llegada,
            'hora_salida_sim': salida,
        })


def generador(env, puerto, datos):
    for _, fila in datos.iterrows():
        tiempo = float(fila['hora_sim'])
        yield env.timeout(max(0.0, tiempo - env.now))
        env.process(puerto.atender_buque(fila))


def main():
    datos = cargar_escalas()
    env = simpy.Environment()
    puerto = Puerto(env)
    env.process(generador(env, puerto, datos))
    env.run()  # Termina cuando salen todos los buques del periodo historico.

    r = pd.DataFrame(puerto.resultados).sort_values('fecha_hora_arribo')
    r.to_csv(SALIDA / 'resultados_por_buque.csv', index=False)

    # Solo se usan para denominadores las horas observadas entre la primera
    # llegada y la ultima salida; son metricas preliminares, sin warm-up.
    horizonte_h = max(float(datos['hora_sim'].max()), float(r['hora_salida_sim'].max()))
    resumen = {
        'escalas_atendidas': len(r),
        'espera_rada_promedio_h': r['espera_rada_h'].mean(),
        'operacion_promedio_h': r['operacion_h'].mean(),
        'estadia_promedio_h': r['estadia_total_h'].mean(),
        'porcentaje_estadias_menores_igual_16h': 100 * (r['estadia_total_h'] <= 16).mean(),
        'utilizacion_bermas_pct': 100 * puerto.berma_horas / (BERMAS * horizonte_h),
        'utilizacion_sts_pct': 100 * puerto.sts_horas / (GRUAS_STS * horizonte_h),
        'horizonte_simulado_h': horizonte_h,
    }
    pd.DataFrame([resumen]).to_csv(SALIDA / 'indicadores_resumen.csv', index=False)

    print('\n=== CODIGO 5: SIMULADOR BASE SIMPY ===')
    print('Escalas atendidas:', len(r))
    print('Espera media en rada: {:.3f} h'.format(resumen['espera_rada_promedio_h']))
    print('Operacion media: {:.3f} h'.format(resumen['operacion_promedio_h']))
    print('Estadia media: {:.3f} h'.format(resumen['estadia_promedio_h']))
    print('Escalas con estadia <= 16 h: {:.2f}%'.format(resumen['porcentaje_estadias_menores_igual_16h']))
    print('Utilizacion de bermas: {:.2f}%'.format(resumen['utilizacion_bermas_pct']))
    print('Utilizacion de STS: {:.2f}%'.format(resumen['utilizacion_sts_pct']))
    print('\nResultados guardados en:', SALIDA)
    print('NOTA: simulacion inicial con llegadas historicas, sin turnos, patio ni validacion.')
    print('El intervalo extraordinario de 636 h se conserva y se estudiara antes de validar.')


if __name__ == '__main__':
    main()
