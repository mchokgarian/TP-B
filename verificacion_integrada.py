# TP2b Puerto Sur Grupo 11 - Parte 19: verificacion del simulador integrado
# No modifica el simulador_final.py ni sustituye la validacion empirica.
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
import simpy
import simulador_final as sf

BASE = Path(__file__).resolve().parent
OUT = BASE / 'Resultados_Verificacion_Integrada'
OUT.mkdir(exist_ok=True)
pruebas = []

def check(nombre, condicion, detalle=''):
    pruebas.append({'control':nombre, 'estado':'OK' if condicion else 'FALLO', 'detalle':str(detalle)})
    if not condicion:
        print('  FALLO:',nombre,detalle,flush=True)

def pendiente(nombre, detalle):
    pruebas.append({'control':nombre, 'estado':'PENDIENTE', 'detalle':detalle})

def validar_resultados_previos():
    ruta = sf.SALIDA / 'comparacion.csv'
    if not ruta.is_file():
        pendiente('Resultados de siete escenarios','Ejecutar primero simulador_final.py; falta comparacion.csv')
        return
    t = pd.read_csv(ruta)
    check('Siete escenarios presentes',set(sf.ESCENARIOS[i]['nombre'] for i in range(len(sf.ESCENARIOS))).issubset(set(t['escenario'])))
    for _,f in t.iterrows():
        n=f['escenario']
        check(n+': conservacion buques',int(f['ingresadas'])==int(f['finalizadas'])+int(f['pendientes']))
        for c in ['ocupacion_maxima_patio_pct','tiempo_sobre_75_pct','utilizacion_bermas_pct','utilizacion_sts_pct']:
            check(n+': '+c, -1e-7 <= float(f[c]) <= 100.00001, f[c])
        if int(f['pendientes']):
            pendiente(n+': indicadores censurados',f"{int(f['pendientes'])} buques sin finalizar; medias solo para finalizados")
        archivo=sf.SALIDA/n/'buques_finalizados.csv'
        if not archivo.exists():
            check(n+': archivo de buques',False,'No se encontro buques_finalizados.csv')
            continue
        b=pd.read_csv(archivo)
        check(n+': filas finales coinciden',len(b)==int(f['finalizadas']))
        check(n+': IDs unicos',not b['id_escala'].duplicated().any())
        for col in ['estadia_total_h','espera_rada_h','operacion_efectiva_h','operacion_transcurrida_h','horas_sts','espera_patio_h','espera_tractores_fin_h']:
            check(n+': '+col+' finito y no negativo',bool(np.isfinite(b[col]).all() and (b[col]>=-1e-8).all()))
        check(n+': estadia >= espera rada',bool((b['estadia_total_h']+1e-7>=b['espera_rada_h']).all()))
        check(n+': estadia >= operacion transcurrida',bool((b['estadia_total_h']+1e-7>=b['operacion_transcurrida_h']).all()))
        check(n+': carga STS <= 2 simultaneas por buque (indicador)',bool(b['uso_dos_gruas'].dropna().isin([0,1,True,False]).all()),'No prueba concurrencia real: se verifica en monitoreo')
        dif=abs(float(f['estadia_media_finalizadas_h'])-float(b['estadia_total_h'].mean()))
        check(n+': media concuerda con CSV',dif<1e-6, dif)

def monitorear(env,p):
    # Muestreo periodico: detecta violaciones durante la simulacion,
    # pero no eventos entre muestreos (no constituye una prueba exhaustiva).
    while True:
        ocup=p.ocupacion
        libre=p.slots_libres.level
        if not (0<=ocup<=sf.CAPACIDAD_PATIO):
            check('Ocupacion dentro de capacidad (muestreo)',False,f't={env.now}, ocup={ocup}')
        if abs(ocup+libre-sf.CAPACIDAD_PATIO)>1e-7:
            check('Balance slots (muestreo)',False,f't={env.now}, ocup={ocup}, libres={libre}')
        if not (0<=p.bermas.count<=p.bermas.capacity):
            check('Capacidad bermas (muestreo)',False,env.now)
        if not (0<=p.gruas.level<=p.gruas.capacity):
            check('Capacidad gruas (muestreo)',False,env.now)
        if not (0<=p.tractores.count<=p.tractores.capacity):
            check('Capacidad tractores (muestreo)',False,env.now)
        if not (0<=p.tractores_noche.count<=p.tractores_noche.capacity):
            check('Capacidad tractores noche (muestreo)',False,env.now)
        if not (0<=p.puertas.count<=p.puertas.capacity):
            check('Capacidad puertas (muestreo)',False,env.now)
        yield env.timeout(.5)

def verificar_dinamicamente():
    datos,origen=sf.cargar_datos()
    datos=datos.iloc[:300].copy()
    config=dict(sf.ESCENARIOS[0])
    sf.BERMAS=int(config['bermas']);sf.STS_TOTALES=int(config['sts'])
    sf.OCUPACION_INICIAL=int(config['ocupacion_inicial'])
    sf.RNG.bit_generator.state=np.random.default_rng(sf.SEMILLA).bit_generator.state
    env=simpy.Environment()
    p=sf.PuertoFinal(env,origen,config)
    env.process(sf.llegadas(env,p,datos))
    monitor=env.process(monitorear(env,p))
    env.run(until=float(datos['hora_sim'].max())+14*24)
    p.actualizar_patio()
    monitor.interrupt() if monitor.is_alive else None
    b=pd.DataFrame(p.registros)
    check('Muestra de 300: llegaron y finalizaron',len(b)==len(datos),f'{len(b)}/{len(datos)}')
    check('Muestra: ocupacion historial valida',all(0<=r['ocupacion']<=sf.CAPACIDAD_PATIO for r in p.historial_patio))
    check('Muestra: conservacion inventario final',abs(p.ocupacion + p.slots_libres.level - sf.CAPACIDAD_PATIO)<1e-8)
    check('Muestra: viajes de tractores contados',p.viajes_realizados>=0)
    check('Muestra: camiones <= programados',len(p.camiones)<=p.lotes_programados)
    check('Muestra: ninguna estadia negativa',bool((b['estadia_total_h']>=0).all()))
    # La ausencia de bloqueos en el caso base no es un fallo.
    check('Muestra base: bloqueos registrados no negativos',p.esperas_por_slots_h>=0)
    check('Muestra base: horas congestionadas no negativas',p.horas_sts_congestionadas>=0)


def verificar_casos_forzados(origen):
    """Pruebas de mecanismos con condiciones artificiales explícitas.

    Son pruebas de verificación de lógica, NO validación con datos observados.
    """
    print('Probando bloqueo de slots y penalizacion por congestion...', flush=True)
    anterior=sf.OCUPACION_INICIAL
    try:
        # Solo quedan cinco slots disponibles; se piden diez. El propio
        # inventario inicial debe liberar capacidad después de 1 hora.
        sf.OCUPACION_INICIAL=sf.CAPACIDAD_PATIO-5
        cfg=dict(sf.ESCENARIOS[0])
        cfg['ocupacion_inicial']=sf.OCUPACION_INICIAL
        env=simpy.Environment()
        p=sf.PuertoFinal(env,origen,cfg)
        tarea=env.process(p.ingresar_contenedores(10))
        env.run(until=1.5)
        check('Slots forzados: operacion termina tras liberacion',tarea.triggered,
              f'finalizada={tarea.triggered}')
        if tarea.triggered and tarea.ok:
            check('Slots forzados: hubo espera real',float(tarea.value)>0.5,
                  f'espera={float(tarea.value):.4f} h')
            check('Slots forzados: inventario conservado',
                  abs(p.ocupacion+p.slots_libres.level-sf.CAPACIDAD_PATIO)<1e-7)
        elif tarea.triggered:
            check('Slots forzados: operacion sin excepcion',False,str(tarea.value))
        else:
            pendiente('Bloqueo de slots bajo prueba extrema',
                      'La solicitud no termino en el horizonte; inspeccionar proceso')

        # Se mantiene una sola grúa disponible para aislar la penalizacion
        # y evitar comparar asignaciones de 1 vs 2 gruas.
        def medir_operacion(penalizacion):
            sf.OCUPACION_INICIAL=2400
            conf=dict(sf.ESCENARIOS[0]);conf['ocupacion_inicial']=2400
            conf['penalizacion']=penalizacion
            e=simpy.Environment(); port=sf.PuertoFinal(e,origen,conf)
            port.origen_h=12.0  # turno diurno; comienzo fuera del relevo
            # Tomamos dos STS de las tres; queda solo una para la operacion.
            e.run(until=port.gruas.get(2))
            proceso=e.process(port.operar({'mov_dry':28.,'mov_reefer':0.,'mov_oog':0.}))
            e.run(until=proceso)
            return e.now,port.horas_sts_congestionadas,port.horas_adicionales_penalizacion
        t0,h0,d0=medir_operacion(0.)
        t15,h15,d15=medir_operacion(.15)
        check('Congestion forzada: STS registra horas congestionadas',h15>0,
              f'horas={h15:.4f}')
        check('Congestion forzada: penalizacion alarga operacion',t15>t0+1e-6,
              f'base={t0:.4f} h; penalizada={t15:.4f} h')
        check('Congestion forzada: demora agregada positiva',d15>0,
              f'horas adicionales={d15:.4f}')
    except Exception as exc:
        check('Pruebas forzadas: ejecucion sin excepcion',False,repr(exc))
    finally:
        sf.OCUPACION_INICIAL=anterior


def revisar_restricciones_pendientes():
    # Este modulo no reescribe el simulador ni considera una prueba
    # documental equivalente a una auditoria de eventos.
    pendiente('STS por buque: auditoria temporal',
              'Hace falta registrar por buque cada adquisicion y devolucion de STS, no solo uso_dos_gruas.')
    pendiente('Turno nocturno y relevos: auditoria temporal',
              'Hace falta un registro cronologico de asignaciones STS y cambios de turno.')
    pendiente('Cruceros: integracion',
              'Los cruceros utilizan otro simulador; verificar compatibilidad de recursos antes de integrarlos.')
    pendiente('Validacion empirica externa',
              'Sin salidas reales y tiempos reales de permanencia no puede contrastarse la estadia simulada.')


def main():
    print('=== PARTE 19: VERIFICACION INTEGRADA COMPLETA ===',flush=True)
    validar_resultados_previos()
    if not sf.ARCHIVO.exists():
        pendiente('Pruebas dinamicas','Falta PuertoSur_escalas_G11.csv junto a los scripts')
    else:
        print('Ejecutando muestra dinamica (300 buques)...',flush=True)
        verificar_dinamicamente()
        _, origen=sf.cargar_datos()
        verificar_casos_forzados(origen)
    revisar_restricciones_pendientes()
    t=pd.DataFrame(pruebas)
    t.to_csv(OUT/'controles_verificacion.csv',index=False)
    resumen=t['estado'].value_counts().to_dict()
    (OUT/'resumen_verificacion.json').write_text(json.dumps(resumen,indent=2),encoding='utf8')
    pendientes=t.loc[t['estado']=='PENDIENTE',['control','detalle']]
    pendientes.to_csv(OUT/'pendientes_documentados.csv',index=False)
    print('\n=== RESULTADOS ===')
    for k in ['OK','FALLO','PENDIENTE']:
        print(k+':',resumen.get(k,0))
    print('Archivos:',OUT)
    print('No convertir los PENDIENTES en OK sin evidencia adicional.')
    if resumen.get('FALLO',0):
        raise SystemExit('Se detectaron fallos: revisar controles_verificacion.csv')

if __name__=='__main__':
    main()
