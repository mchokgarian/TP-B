# ============================================================
# TP2b - PUERTO SUR - GRUPO 11
# CODIGO 13: VERIFICACION DE RESULTADOS DEL SIMULADOR
# Ejecutar: python verificacion_simulador.py
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SALIDA = BASE / 'Resultados_Verificacion'
SALIDA.mkdir(exist_ok=True)
CSV = BASE / 'PuertoSur_escalas_G11.csv'
TOL = 1e-5

PRUEBAS = []

def prueba(bloque, nombre, condicion, detalle=''):
    estado = 'OK' if bool(condicion) else 'FALLA'
    PRUEBAS.append(dict(bloque=bloque, prueba=nombre, estado=estado, detalle=str(detalle)))
    print(f'[{estado}] {bloque}: {nombre}' + (f' | {detalle}' if detalle else ''))


def leer_csv(ruta, obligatorio=False):
    if not ruta.exists():
        prueba('Archivos', f'Existe {ruta.relative_to(BASE)}', not obligatorio, 'No encontrado' + (' (obligatorio)' if obligatorio else ' (opcional)'))
        return None
    try:
        return pd.read_csv(ruta)
    except Exception as exc:
        prueba('Archivos', f'Lectura {ruta.name}', False, repr(exc))
        return None


def verificar_buques(nombre, carpeta, referencia):
    r = leer_csv(BASE / carpeta / 'resultados_por_buque.csv')
    if r is None:
        return
    prueba(nombre, 'IDs únicos', 'id_escala' in r and not r['id_escala'].duplicated().any())
    prueba(nombre, 'Todas las escalas atendidas', len(r) == len(referencia), f'{len(r)}/{len(referencia)}')
    if 'id_escala' in r:
        prueba(nombre, 'IDs corresponden al histórico', set(r['id_escala']) == set(referencia['id_escala']))
    for col in ['estadia_total_h', 'espera_rada_h']:
        if col in r:
            x = pd.to_numeric(r[col], errors='coerce')
            prueba(nombre, f'{col} finito y no negativo', np.isfinite(x).all() and (x >= -TOL).all())
    if {'estadia_total_h', 'espera_rada_h'}.issubset(r.columns):
        prueba(nombre, 'Estadía >= espera en rada', (r['estadia_total_h'] + TOL >= r['espera_rada_h']).all())
    if {'estadia_total_h', 'espera_rada_h', 'maniobras_h', 'operacion_transcurrida_h'}.issubset(r.columns):
        diferencia = r['estadia_total_h'] - (r['espera_rada_h'] + r['maniobras_h'] + r['operacion_transcurrida_h'])
        otros = r['espera_patio_h'] if 'espera_patio_h' in r.columns else 0
        diferencia = diferencia - otros
        prueba(nombre, 'Balance de tiempos (sin intervalos solapados)', np.isfinite(diferencia).all() and (diferencia >= -1e-3).all(), f'mínimo residual={diferencia.min():.4f} h')
    if 'uso_dos_gruas' in r:
        prueba(nombre, 'Indicador de 2 grúas válido', r['uso_dos_gruas'].dropna().isin([True, False, 0, 1, 'True', 'False']).all())


def main():
    ref = leer_csv(CSV, obligatorio=True)
    if ref is None:
        print('Falta el CSV histórico. No se puede continuar.')
        return
    prueba('Histórico', 'IDs únicos', ref['id_escala'].is_unique)
    prueba('Histórico', 'Movimientos >= 0', (ref[['mov_dry','mov_reefer','mov_oog']] >= 0).all().all())

    carpetas = [
        ('5 Base', 'Resultados_Simulador_Base'),
        ('6 Turnos', 'Resultados_Simulador_Turnos'),
        ('7 Dos grúas', 'Resultados_Dos_Gruas'),
        ('8 Patio', 'Resultados_Patio'),
        ('9 Tractores', 'Resultados_Tractores'),
    ]
    for nombre, carpeta in carpetas:
        verificar_buques(nombre, carpeta, ref)

    # El código 10 puede almacenar solamente indicadores agregados.
    c10 = leer_csv(BASE / 'Resultados_Camiones' / 'indicadores_resumen.csv')
    if c10 is not None and len(c10):
        if 'escalas_atendidas' in c10:
            prueba('10 Camiones', 'Total de escalas', int(c10.iloc[0]['escalas_atendidas']) == len(ref))
        for campo in ['impuntualidad_pct', 'fuera_ventana_pct', 'ocupacion_maxima_patio_pct']:
            if campo in c10:
                n = float(c10.iloc[0][campo])
                prueba('10 Camiones', f'{campo} entre 0 y 100', 0 <= n <= 100)
        if 'lotes_camiones_pendientes' in c10:
            prueba('10 Camiones', 'Sin lotes pendientes', int(c10.iloc[0]['lotes_camiones_pendientes']) == 0)

    e = leer_csv(BASE / 'Resultados_Escenarios' / 'comparacion_escenarios.csv', obligatorio=True)
    if e is not None:
        prueba('12 Escenarios', '12 combinaciones', len(e) == 12, f'{len(e)} filas')
        cols = {'configuracion', 'demanda', 'escalas_ingresadas', 'escalas_finalizadas', 'escalas_pendientes'}
        prueba('12 Escenarios', 'Columnas esenciales presentes', cols.issubset(e.columns))
        if cols.issubset(e.columns):
            prueba('12 Escenarios', 'Ingreso = finalizadas + pendientes', (e['escalas_ingresadas'] == e['escalas_finalizadas'] + e['escalas_pendientes']).all())
            prueba('12 Escenarios', 'Finalizadas no exceden ingresos', (e['escalas_finalizadas'] <= e['escalas_ingresadas']).all())
            prueba('12 Escenarios', 'Se identifican escenarios censurados', True, f'{(e.escalas_pendientes > 0).sum()} con pendientes')
            if 'estadia_media_finalizadas_h' in e:
                saturados = e.loc[e['escalas_pendientes'] > 0, ['configuracion', 'demanda', 'escalas_pendientes', 'estadia_media_finalizadas_h']]
                saturados.to_csv(SALIDA / 'escenarios_con_pendientes.csv', index=False)

    # Advertencias metodológicas que no pueden evaluarse con agregados.
    advertencias = [
        'Los controles anteriores comprueban coherencia de archivos/resultados; NO demuestran que la lógica interna de SimPy sea correcta.',
        'Verificar en código/trazas: máximo 1 buque por berma; máximo 3 STS instaladas (4 en escenarios); máximo 2 STS por buque.',
        'Verificar en código/trazas: dotación nocturna, pausas de relevo, alistamiento standby y que ninguna STS esté asignada dos veces.',
        'Verificar conservación de slots y flujo real de descarga/carga, tractores y camiones. El CSV no contiene zarpadas históricas para validar estadías.',
        'La demanda +35%/+100% fue generada comprimiendo tiempos de arribos; el período excepcional de 636 h debe tratarse en sensibilidad.',
        'No comparar promedios de estadía como si fueran completos cuando hay escalas pendientes (censura).',
        'Conservar una lista de supuestos sin fuente (flota/capacidad de tractores, ventanas, ocupación inicial, permanencias y reserva de cruceros).',
    ]
    (SALIDA / 'pendientes_validacion.txt').write_text('\n'.join('- ' + x for x in advertencias), encoding='utf-8')
    for x in advertencias:
        print('PENDIENTE:', x)

    tabla = pd.DataFrame(PRUEBAS)
    tabla.to_csv(SALIDA / 'reporte_verificacion.csv', index=False)
    fallas = int((tabla.estado == 'FALLA').sum())
    print(f'\n=== CODIGO 13: VERIFICACION ===\nControles: {len(tabla)} | Fallas: {fallas}')
    print('Resultados:', SALIDA)
    print('ATENCION: sin fallas NO equivale a modelo validado.')


if __name__ == '__main__':
    main()
