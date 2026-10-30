# TP2b - Puerto Sur - Grupo 11 | Codigo 11: cruceros y reserva de berma
# Requiere simulador_camiones.py, simulador_tractores.py (corregido) y simulador_patio.py
from pathlib import Path
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt

from simulador_camiones import Puerto as PuertoCamiones
from simulador_patio import cargar_datos, TASA, FRACCION_DESCARGA, LOTE_DESCARGA, RNG as RNG_MODELO

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Cruceros'
SALIDA.mkdir(exist_ok=True)

# SUPUESTOS EXPLORATORIOS (NO SON DATOS OBSERVADOS)
MESES_FRUTA = (12, 1, 2)  # ajustar cuando se confirme temporada
DIAS_CRUCERO = (1, 4)     # martes y viernes: 2 por semana
HORA_PROGRAMADA = 9.0    # arribo matutino
PREAVISO_RESERVA_H = 36.0
FIN_VENTANA_RESERVA_H = 40.0   # desde hora programada
MARGEN_FEEDER_H = 2.0
INGRESO_CRUCERO_USD = 60000.0  # dato de la consigna
SEMILLA_CRUCEROS = 511


def generar_cruceros(inicio, fin):
    dias = pd.date_range(inicio.normalize(), fin.normalize(), freq='D')
    filas = []
    i = 0
    for d in dias:
        if d.month not in MESES_FRUTA or d.weekday() not in DIAS_CRUCERO:
            continue
        programado = d + pd.Timedelta(hours=HORA_PROGRAMADA)
        if programado < inicio or programado > fin:
            continue
        i += 1
        filas.append({'id_crucero': f'CRU-{i:04d}',
                      'programado_h': (programado - inicio).total_seconds()/3600})
    return pd.DataFrame(filas)


class Puerto(PuertoCamiones):
    def __init__(self, env, origen, cruceros, politica, rng):
        super().__init__(env, origen)
        self.bermas_fisicas = simpy.FilterStore(env, capacity=3)
        self.bermas_fisicas.items.extend([0, 1, 2])
        self.politica_cruceros = politica
        self.rng_cruceros = rng
        self.programados = sorted(cruceros['programado_h'].tolist())
        self.cruceros_resultado = []
        self.horas_berma_crucero = 0.0

    def es_apto_berma(self, numero, trabajo_h):
        if self.politica_cruceros == 'sin_reserva' or numero != 2:
            return True
        # Ventana de resguardo para la berma 3. Fuera de ella
        # la berma puede ser utilizada por buques feeder.
        ahora = self.env.now
        for t in self.programados:
            if t - PREAVISO_RESERVA_H - trabajo_h <= ahora < t + FIN_VENTANA_RESERVA_H:
                return False
        return True

    def atender(self, fila):
        llegada = self.env.now
        trabajo = sum(float(fila[c]) / v for c, v in TASA.items())
        duracion_previsible = trabajo + 2 + MARGEN_FEEDER_H
        solicitud = self.bermas_fisicas.get(
            lambda b: self.es_apto_berma(b, duracion_previsible)
        )
        numero_berma = yield solicitud
        espera_rada = self.env.now - llegada
        ingreso = self.env.now
        try:
            maniobras = RNG_MODELO.uniform(1, 2)
            yield self.env.timeout(maniobras / 2)
            movimientos = int(sum(int(fila[c]) for c in TASA))
            descarga = int(round(movimientos * FRACCION_DESCARGA))
            faltantes = descarga
            lotes = []
            while faltantes:
                n = min(LOTE_DESCARGA, faltantes)
                lotes.append(n)
                faltantes -= n
            if not lotes:
                lotes = [0]
            inicio_op = self.env.now
            operacion_efectiva = 0.0
            horas_sts = 0.0
            uso_dos = False
            tareas = []
            for lote in lotes:
                proporcion = lote / descarga if descarga else 1.0
                subcarga = {c: float(fila[c]) * proporcion for c in TASA}
                activo, hs, dos = yield from self.operar(subcarga)
                operacion_efectiva += activo
                horas_sts += hs
                uso_dos = uso_dos or dos
                if lote:
                    tareas.append(self.env.process(self.transportar_lote(lote)))
            fin_gruas = self.env.now
            if tareas:
                demoras = yield self.env.all_of(tareas)
                espera_tractores_fin = max(0, self.env.now - fin_gruas)
                espera_patio = sum(float(demoras[t] or 0) for t in tareas)
            else:
                espera_tractores_fin = 0.0
                espera_patio = 0.0
            fin_op = self.env.now
            yield self.env.timeout(maniobras / 2)
            salida = self.env.now
            self.uso_bermas += salida - ingreso
        finally:
            yield self.bermas_fisicas.put(numero_berma)
        self.registros.append({
            'id_escala': fila['id_escala'],
            'servicio': fila['servicio'],
            'fecha_hora_arribo': fila['fecha_hora_arribo'],
            'berma': numero_berma + 1,
            'espera_rada_h': espera_rada,
            'espera_patio_h': espera_patio,
            'espera_tractores_fin_h': espera_tractores_fin,
            'operacion_efectiva_h': operacion_efectiva,
            'operacion_transcurrida_h': fin_op - inicio_op,
            'horas_sts': horas_sts,
            'uso_dos_gruas': uso_dos,
            'contenedores_descargados': descarga,
            'estadia_total_h': salida - llegada,
            'hora_salida_sim': salida,
        })

    def atender_crucero(self, registro):
        programado = registro['programado_h']
        desviacion = self.rng_cruceros.uniform(-3, 3)
        llegada = max(0, programado + desviacion)
        yield self.env.timeout(max(0, llegada - self.env.now))
        if self.politica_cruceros == 'con_reserva':
            solicitud = self.bermas_fisicas.get(lambda b: b == 2)
        else:
            solicitud = self.bermas_fisicas.get()
        berma = yield solicitud
        espera = self.env.now - llegada
        ingreso = self.env.now
        permanencia = self.rng_cruceros.uniform(28, 36)
        try:
            yield self.env.timeout(permanencia)
            salida = self.env.now
            self.horas_berma_crucero += salida - ingreso
        finally:
            yield self.bermas_fisicas.put(berma)
        self.cruceros_resultado.append({
            'id_crucero': registro['id_crucero'],
            'programado_h': programado,
            'arribo_h': llegada,
            'berma': berma + 1,
            'espera_rada_h': espera,
            'ocupacion_berma_h': permanencia,
            'estadia_total_h': salida - llegada,
            'hora_salida_sim': salida,
            'ingreso_usd': INGRESO_CRUCERO_USD,
        })


def llegadas_buques(env, puerto, datos):
    for i, (_, fila) in enumerate(datos.iterrows(), start=1):
        yield env.timeout(max(0, float(fila['hora_sim']) - env.now))
        env.process(puerto.atender(fila))
        if i % 1000 == 0:
            print(f'Ingresaron {i} de {len(datos)} buques feeder', flush=True)


def ejecutar(politica, datos, origen, cruceros):
    # Mismas perturbaciones de cruceros en ambas politicas.
    RNG_MODELO.bit_generator.state = np.random.default_rng(51).bit_generator.state
    env = simpy.Environment()
    puerto = Puerto(env, origen, cruceros, politica,
                    np.random.default_rng(SEMILLA_CRUCEROS))
    env.process(llegadas_buques(env, puerto, datos))
    for _, registro in cruceros.iterrows():
        env.process(puerto.atender_crucero(registro))
    env.run(until=float(datos['hora_sim'].max()) + 1000)
    barcos = pd.DataFrame(puerto.registros)
    cruceros_out = pd.DataFrame(puerto.cruceros_resultado)
    if len(barcos) != len(datos) or len(cruceros_out) != len(cruceros):
        raise RuntimeError(f'{politica}: feeder {len(barcos)}/{len(datos)}, '
                           f'cruceros {len(cruceros_out)}/{len(cruceros)}; revisar estabilidad.')
    carpeta = SALIDA / politica
    carpeta.mkdir(exist_ok=True)
    barcos.to_csv(carpeta / 'resultados_feeder.csv', index=False)
    cruceros_out.to_csv(carpeta / 'resultados_cruceros.csv', index=False)
    pd.DataFrame(puerto.camiones).to_csv(carpeta / 'resultados_camiones.csv', index=False)
    puerto.actualizar_patio()
    horizonte = max(barcos['hora_salida_sim'].max(), cruceros_out['hora_salida_sim'].max())
    return {
        'politica': politica,
        'buques_feeder': len(barcos),
        'cruceros': len(cruceros_out),
        'espera_media_feeder_h': barcos['espera_rada_h'].mean(),
        'estadia_media_feeder_h': barcos['estadia_total_h'].mean(),
        'feeder_cumple_16h_pct': 100 * (barcos['estadia_total_h'] <= 16).mean(),
        'espera_media_crucero_h': cruceros_out['espera_rada_h'].mean(),
        'cruceros_sin_espera_pct': 100 * (cruceros_out['espera_rada_h'] < 1e-8).mean(),
        'utilizacion_bermas_pct': 100 * (puerto.uso_bermas + puerto.horas_berma_crucero)/(3*horizonte),
        'ingreso_bruto_cruceros_usd': cruceros_out['ingreso_usd'].sum(),
    }


def main():
    datos, origen = cargar_datos()
    cruceros = generar_cruceros(origen, datos['fecha_hora_arribo'].max())
    print('Cruceros programados:', len(cruceros), flush=True)
    resultados = []
    for politica in ('sin_reserva', 'con_reserva'):
        print('\nEjecutando escenario:', politica, flush=True)
        r = ejecutar(politica, datos, origen, cruceros)
        resultados.append(r)
        print('Estadia media feeder:', round(r['estadia_media_feeder_h'], 3), 'h')
        print('Espera media crucero:', round(r['espera_media_crucero_h'], 3), 'h')
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(SALIDA / 'comparacion_politicas.csv', index=False)
    print('\n=== CODIGO 11: COMPARACION DE CRUCEROS ===')
    print(tabla.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    plt.figure(figsize=(8, 4))
    plt.bar(tabla['politica'], tabla['estadia_media_feeder_h'])
    plt.axhline(16, linestyle='--', label='Objetivo feeder 16 h')
    plt.ylabel('Estadia media feeder (horas)')
    plt.title('Cruceros: efecto de reservar una berma')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SALIDA / '01_comparacion_estadia_feeder.png', dpi=180)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.bar(tabla['politica'], tabla['espera_media_crucero_h'])
    plt.ylabel('Espera media crucero (horas)')
    plt.tight_layout()
    plt.savefig(SALIDA / '02_comparacion_espera_cruceros.png', dpi=180)
    plt.close()
    print('\nResultados guardados en:', SALIDA)
    print('NOTA: temporada (dic-feb), dias, horario y preaviso de reserva son SUPUESTOS.')
    print('La reserva es aproximada (resguardo anticipado de la berma 3).')


if __name__ == '__main__':
    main()
