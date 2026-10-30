# PUERTO SUR GRUPO 11 - SIMULADOR INTEGRADO EXPERIMENTAL
# Integra Codigos 8, 9, 10, 16 y 17 de la carpeta del usuario.
# Por ahora NO es un modelo empiricamente validado.
# Requiere: PuertoSur_escalas_G11.csv, pip install simpy pandas numpy matplotlib
# ==============================================================================
# PUERTO SUR GRUPO 11 - SIMULADOR INTEGRADO CON PANEL INTERACTIVO
# Integra la lógica de turnos, patio, tractores, camiones y congestión
# con una interfaz gráfica basada en Tkinter.
# ==============================================================================
# PUERTO SUR GRUPO 11 - SIMULADOR CON ANIMACIÓN ULTRA RÁPIDA EN TIEMPO REAL
# Incluye aceleración por lotes (batching) y control dinámico de velocidad.
# ==============================================================================
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import simpy

# --- RUTAS DE ACCESO ---
BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "PuertoSur_escalas_G11.csv"
SALIDA = BASE / "Resultados_Simulador_Final"
SALIDA.mkdir(exist_ok=True)

# --- PARÁMETROS OPERATIVOS ---
CAPACIDAD_PATIO = 3000
UMBRAL_CONGESTION = 0.75
PERMANENCIA_MEDIA_H = 72.0
FRACCION_DESCARGA = 0.50
LOTE_DESCARGA = 10
TRACTORES_DIA = 4
TRACTORES_NOCHE = 2
CONTENEDORES_POR_VIAJE = 5
TIEMPO_VIAJE_MIN = 15
TASA = {"mov_dry": 28.0, "mov_reefer": 21.0, "mov_oog": 12.0}

ESCENARIOS_PREDEFINIDOS = {
    "base": dict(
        demanda=1.0,
        bermas=3,
        sts=3,
        ocupacion_inicial=1500,
        puertas=3,
        impuntuales=0.25,
        atraso_extra_h=4.0,
        permanencia_h=72.0,
        penalizacion=0.0,
    ),
    "cuarta_sts": dict(
        demanda=1.0,
        bermas=3,
        sts=4,
        ocupacion_inicial=1500,
        puertas=3,
        impuntuales=0.25,
        atraso_extra_h=4.0,
        permanencia_h=72.0,
        penalizacion=0.0,
    ),
    "cuarta_berma": dict(
        demanda=1.0,
        bermas=4,
        sts=3,
        ocupacion_inicial=1500,
        puertas=3,
        impuntuales=0.25,
        atraso_extra_h=4.0,
        permanencia_h=72.0,
        penalizacion=0.0,
    ),
    "ambas": dict(
        demanda=1.0,
        bermas=4,
        sts=4,
        ocupacion_inicial=1500,
        puertas=3,
        impuntuales=0.25,
        atraso_extra_h=4.0,
        permanencia_h=72.0,
        penalizacion=0.0,
    ),
    "congestion_15": dict(
        demanda=1.0,
        bermas=3,
        sts=3,
        ocupacion_inicial=2100,
        puertas=2,
        impuntuales=0.50,
        atraso_extra_h=24.0,
        permanencia_h=96.0,
        penalizacion=0.15,
    ),
}


# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
def cargar_datos():
    if not ARCHIVO.exists():
        raise FileNotFoundError(
            f"Falta el archivo {ARCHIVO.name} en el directorio."
        )

    df = pd.read_csv(ARCHIVO, parse_dates=["fecha_hora_arribo"])
    df = df.sort_values("fecha_hora_arribo").reset_index(drop=True)

    columnas = [
        "id_escala",
        "fecha_hora_arribo",
        "servicio",
        "mov_dry",
        "mov_reefer",
        "mov_oog",
    ]
    if df[columnas].isna().any().any():
        raise ValueError("Hay datos nulos en el CSV.")

    origen = df["fecha_hora_arribo"].iloc[0]
    df["hora_sim"] = (
        df["fecha_hora_arribo"] - origen
    ).dt.total_seconds() / 3600.0
    return df, origen


# ==============================================================================
# MODELO SIMPY VINCULADO AL TABLERO GRÁFICO
# ==============================================================================
class PuertoVisual:

    def __init__(self, env, origen, config, rng, monitor):
        self.env = env
        self.config = config
        self.rng = rng
        self.monitor = monitor

        self.origen_h = (
            origen.hour + origen.minute / 60.0 + origen.second / 3600.0
        )
        self.n_bermas = int(config["bermas"])
        self.bermas = simpy.Resource(env, capacity=self.n_bermas)
        self.gruas = simpy.Container(
            env, capacity=int(config["sts"]), init=int(config["sts"])
        )

        self.ocupacion_inicial = int(config["ocupacion_inicial"])
        self.slots_libres = simpy.Container(
            env,
            capacity=CAPACIDAD_PATIO,
            init=CAPACIDAD_PATIO - self.ocupacion_inicial,
        )

        self.ocupacion = self.ocupacion_inicial
        env.process(self.liberar_stock_inicial())

        self.registros = []
        self.uso_bermas = 0.0
        self.uso_gruas = 0.0

        self.relevos = {}
        self.recursos_noche = {}
        self.ocupacion_maxima = self.ocupacion_inicial
        self.area_ocupacion = 0.0
        self.tiempo_congestion = 0.0
        self.ultimo_cambio_patio = 0.0
        self.esperas_patio = []

        self.estado_bermas = [None] * self.n_bermas

        self.tractores = simpy.Resource(env, capacity=TRACTORES_DIA)
        self.tractores_noche = simpy.Resource(env, capacity=TRACTORES_NOCHE)
        self.puertas = simpy.Resource(env, capacity=int(config["puertas"]))
        self.camiones = []

    def calendario(self):
        return self.origen_h + self.env.now

    def turno_actual(self):
        t = self.calendario()
        dia = int(t // 24)
        hora = t % 24
        inicio = dia * 24
        if hora < 8:
            return inicio, inicio + 8, True
        if hora < 16:
            return inicio + 8, inicio + 16, False
        return inicio + 16, inicio + 24, False

    def duracion_relevo(self, inicio):
        if inicio not in self.relevos:
            self.relevos[inicio] = self.rng.uniform(10, 15) / 60.0
        return self.relevos[inicio]

    def esperar_relevo(self):
        inicio, _, _ = self.turno_actual()
        demora = inicio + self.duracion_relevo(inicio) - self.calendario()
        if demora > 1e-9:
            yield self.env.timeout(demora)

    def recurso_nocturno(self, inicio):
        if inicio not in self.recursos_noche:
            recurso = simpy.Container(self.env, capacity=2, init=1)
            self.recursos_noche[inicio] = recurso
            demora = self.rng.uniform(15, 20) / 60.0
            self.env.process(self.activar_standby(recurso, demora))
        return self.recursos_noche[inicio]

    def activar_standby(self, recurso, demora):
        yield self.env.timeout(demora)
        yield recurso.put(1)

    def actualizar_patio(self):
        ahora = self.env.now
        delta = ahora - self.ultimo_cambio_patio
        self.area_ocupacion += self.ocupacion * delta
        if (self.ocupacion / CAPACIDAD_PATIO) > UMBRAL_CONGESTION:
            self.tiempo_congestion += delta
        self.ultimo_cambio_patio = ahora
        self.ocupacion_maxima = max(self.ocupacion_maxima, self.ocupacion)
        self.monitor.notificar_patio(self.ocupacion, CAPACIDAD_PATIO)

    def liberar_stock_inicial(self):
        n_periodos = 72
        base, sobrante = divmod(self.ocupacion_inicial, n_periodos)
        for i in range(n_periodos):
            yield self.env.timeout(PERMANENCIA_MEDIA_H / n_periodos)
            cantidad = base + (1 if i < sobrante else 0)
            self.actualizar_patio()
            self.ocupacion -= cantidad
            yield self.slots_libres.put(cantidad)

    def ingresar_contenedores(self, cantidad):
        if cantidad <= 0:
            return 0.0
        inicio = self.env.now
        yield self.slots_libres.get(cantidad)
        demora = self.env.now - inicio
        self.esperas_patio.append(demora)
        self.actualizar_patio()
        self.ocupacion += cantidad
        self.env.process(self.liberar_contenedores(cantidad))
        return demora

    def liberar_contenedores(self, cantidad):
        c = self.config
        impuntual = self.rng.random() < c["impuntuales"]
        atraso = self.rng.uniform(1, c["atraso_extra_h"]) if impuntual else 0.0

        yield self.env.timeout(c["permanencia_h"] + atraso)
        inicio_cola = self.env.now
        with self.puertas.request() as permiso:
            yield permiso
            espera = self.env.now - inicio_cola
            yield self.env.timeout(10.0 / 60.0)

        self.actualizar_patio()
        self.ocupacion -= cantidad
        yield self.slots_libres.put(cantidad)
        self.camiones.append(
            dict(
                cantidad=cantidad,
                impuntual=impuntual,
                espera_cola_h=espera,
                atraso_h=atraso,
                hora_retiro=self.env.now,
            )
        )

    def _viaje_tractor(self, cantidad):
        inicio_espera = self.env.now
        _, _, noche = self.turno_actual()
        if noche:
            with self.tractores_noche.request() as pn:
                yield pn
                with self.tractores.request() as pdia:
                    yield pdia
                    duracion = TIEMPO_VIAJE_MIN / 60.0
                    yield self.env.timeout(duracion)
        else:
            with self.tractores.request() as pdia:
                yield pdia
                duracion = TIEMPO_VIAJE_MIN / 60.0
                yield self.env.timeout(duracion)
        return (yield from self.ingresar_contenedores(cantidad))

    def transportar_lote(self, cantidad):
        if cantidad <= 0:
            return 0.0
        pendientes = int(cantidad)
        viajes = []
        while pendientes:
            carga = min(CONTENEDORES_POR_VIAJE, pendientes)
            viajes.append(self.env.process(self._viaje_tractor(carga)))
            pendientes -= carga
        valores = yield self.env.all_of(viajes)
        return max(float(valores[v] or 0.0) for v in viajes)

    def operar(self, carga, idx_berma, id_buque):
        trabajo = sum(float(carga[k]) / tasa for k, tasa in TASA.items())
        tiempo_efectivo = 0.0
        horas_sts = 0.0
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
            interferencia = 0.85 if n == 2 else 1.0
            congestion = (self.ocupacion / CAPACIDAD_PATIO) > UMBRAL_CONGESTION
            penalizacion = self.config["penalizacion"] if congestion else 0.0
            velocidad = n * interferencia * (1.0 - penalizacion)

            duracion = min(trabajo / velocidad, max(0.0, fin - self.calendario()))

            self.monitor.actualizar_berma(
                idx_berma, f"Buque {id_buque}\nOperando ({n} STS)", "#0284c7"
            )

            if duracion > 1e-9:
                yield self.env.timeout(duracion)
                trabajo -= duracion * velocidad
                tiempo_efectivo += duracion
                horas_sts += n * duracion
                self.uso_gruas += n * duracion

            yield self.gruas.put(n)
            if noche:
                yield recurso_noche.put(n)
            if duracion <= 1e-9:
                yield self.env.timeout(1e-6)

        return tiempo_efectivo, horas_sts, uso_dos

    def atender(self, fila):
        llegada = self.env.now
        id_buque = fila["id_escala"]

        self.monitor.actualizar_rada(len(self.bermas.queue) + 1)

        with self.bermas.request() as permiso:
            yield permiso
            espera_rada = self.env.now - llegada
            ingreso_berma = self.env.now

            idx_berma = self.estado_bermas.index(None)
            self.estado_bermas[idx_berma] = id_buque
            self.monitor.actualizar_rada(len(self.bermas.queue))

            # Atraque
            maniobras = self.rng.uniform(1, 2)
            self.monitor.actualizar_berma(
                idx_berma, f"Buque {id_buque}\nAtraque...", "#d97706"
            )
            yield self.env.timeout(maniobras / 2.0)

            # Carga y descarga
            movimientos = int(sum(int(fila[c]) for c in TASA))
            cantidad_descarga = int(round(movimientos * FRACCION_DESCARGA))
            restante = cantidad_descarga
            lotes = []
            while restante > 0:
                n = min(LOTE_DESCARGA, restante)
                lotes.append(n)
                restante -= n
            if not lotes:
                lotes = [0]

            inicio_op = self.env.now
            operacion_efectiva = 0.0
            horas_sts = 0.0
            uso_dos = False
            tareas_transporte = []

            for lote in lotes:
                fraccion = (
                    (lote / cantidad_descarga) if cantidad_descarga else 1.0
                )
                subcarga = {c: float(fila[c]) * fraccion for c in TASA}
                activo, hs, dos = yield from self.operar(
                    subcarga, idx_berma, id_buque
                )
                operacion_efectiva += activo
                horas_sts += hs
                uso_dos = uso_dos or dos

                if lote:
                    tareas_transporte.append(
                        self.env.process(self.transportar_lote(lote))
                    )

            if tareas_transporte:
                yield self.env.all_of(tareas_transporte)

            fin_op = self.env.now

            # Desatraque
            self.monitor.actualizar_berma(
                idx_berma, f"Buque {id_buque}\nDesatraque...", "#d97706"
            )
            yield self.env.timeout(maniobras / 2.0)
            salida = self.env.now
            self.uso_bermas += salida - ingreso_berma

            self.estado_bermas[idx_berma] = None
            self.monitor.actualizar_berma(idx_berma, "Libre", "#16a34a")

        self.registros.append(
            {
                "id_escala": id_buque,
                "servicio": fila["servicio"],
                "fecha_hora_arribo": fila["fecha_hora_arribo"],
                "espera_rada_h": espera_rada,
                "operacion_efectiva_h": operacion_efectiva,
                "estadia_total_h": salida - llegada,
                "hora_salida_sim": salida,
            }
        )
        self.monitor.notificar_buque_finalizado(len(self.registros))


def generador_llegadas(env, puerto, datos):
    for row in datos.itertuples(index=False):
        fila = pd.Series(row._asdict())
        yield env.timeout(max(0.0, float(fila["hora_sim"]) - env.now))
        env.process(puerto.atender(fila))


# ==============================================================================
# PANEL DE CONTROL PRINCIPAL
# ==============================================================================
class AppSimulador(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Panel de Control - Puerto Sur G11")
        self.geometry("900x670")
        self.resizable(False, False)

        self.COLOR_BG = "#1e293b"
        self.COLOR_CARD = "#0f172a"
        self.COLOR_TEXT = "#f8fafc"
        self.COLOR_ACCENT = "#38bdf8"
        self.COLOR_BTN = "#10b981"
        self.COLOR_INPUT = "#334155"

        self.configure(bg=self.COLOR_BG)
        self.construir_configurador()

    def construir_configurador(self):
        lbl_titulo = tk.Label(
            self,
            text="⚙ CONFIGURADOR DE ESCENARIOS OPERATIVOS",
            font=("Segoe UI", 15, "bold"),
            bg=self.COLOR_BG,
            fg="#ffffff",
        )
        lbl_titulo.pack(pady=(15, 6))

        frame_top = tk.Frame(self, bg=self.COLOR_BG)
        frame_top.pack(fill="x", padx=25, pady=(0, 10))

        tk.Label(
            frame_top,
            text="Cargar Escenario Predefinido:",
            font=("Segoe UI", 10, "bold"),
            bg=self.COLOR_BG,
            fg=self.COLOR_ACCENT,
        ).pack(side="left", padx=(0, 8))

        self.combo_escenario = ttk.Combobox(
            frame_top,
            values=list(ESCENARIOS_PREDEFINIDOS.keys()),
            state="readonly",
            width=20,
        )
        self.combo_escenario.set("base")
        self.combo_escenario.pack(side="left")
        self.combo_escenario.bind(
            "<<ComboboxSelected>>", self.cargar_escenario_seleccionado
        )

        contenedor = tk.Frame(self, bg=self.COLOR_BG)
        contenedor.pack(fill="both", expand=True, padx=25)

        # Columna Izquierda
        col_izq = tk.LabelFrame(
            contenedor,
            text=" Infraestructura y Demanda ",
            bg=self.COLOR_CARD,
            fg=self.COLOR_ACCENT,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        col_izq.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.var_demanda = self.crear_input(
            col_izq, "Factor Demanda (1.0 = base):", "1.0"
        )
        self.var_bermas = self.crear_input(
            col_izq, "Cantidad de Bermas (1 a 6):", "3"
        )
        self.var_sts = self.crear_input(col_izq, "Grúas STS Totales:", "3")
        self.var_ocupacion = self.crear_input(
            col_izq, "Ocupación Inicial de Patio (TEU):", "1500"
        )

        self.var_semilla_libre = tk.BooleanVar(value=False)
        chk_semilla = tk.Checkbutton(
            col_izq,
            text="Variabilidad libre (sin semilla)",
            variable=self.var_semilla_libre,
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT,
            selectcolor=self.COLOR_INPUT,
            activebackground=self.COLOR_CARD,
            activeforeground="white",
        )
        chk_semilla.pack(anchor="w", pady=(10, 2))

        self.var_semilla = self.crear_input(
            col_izq, "Semilla del generador:", "51"
        )

        # Columna Derecha
        col_der = tk.LabelFrame(
            contenedor,
            text=" Logística Terrestre y Patio ",
            bg=self.COLOR_CARD,
            fg=self.COLOR_ACCENT,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
        )
        col_der.pack(side="right", fill="both", expand=True, padx=(8, 0))

        self.var_puertas = self.crear_input(
            col_der, "Puertas de Camiones:", "3"
        )
        self.var_impuntuales = self.crear_input(
            col_der, "Fracción Impuntuales (0 a 1):", "0.25"
        )
        self.var_atraso = self.crear_input(
            col_der, "Atraso Máximo (horas):", "4.0"
        )
        self.var_permanencia = self.crear_input(
            col_der, "Permanencia Media (horas):", "72.0"
        )
        self.var_penalizacion = self.crear_input(
            col_der, "Penalización Congestión (>75% patio):", "0.0"
        )

        # Botón Iniciar
        btn_iniciar = tk.Button(
            self,
            text="🚀 INICIAR SIMULACIÓN",
            font=("Segoe UI", 12, "bold"),
            bg=self.COLOR_BTN,
            fg="#022c22",
            activebackground="#34d399",
            cursor="hand2",
            relief="flat",
            command=self.abrir_ventana_animacion,
        )
        btn_iniciar.pack(pady=15, ipadx=25, ipady=8)

    def crear_input(self, padre, etiqueta, val_defecto):
        tk.Label(
            padre,
            text=etiqueta,
            bg=self.COLOR_CARD,
            fg=self.COLOR_TEXT,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 1))
        var = tk.StringVar(value=str(val_defecto))
        ent = tk.Entry(
            padre,
            textvariable=var,
            bg=self.COLOR_INPUT,
            fg="white",
            insertbackground="white",
        )
        ent.pack(fill="x", pady=(0, 4))
        return var

    def cargar_escenario_seleccionado(self, event=None):
        clave = self.combo_escenario.get()
        if clave in ESCENARIOS_PREDEFINIDOS:
            d = ESCENARIOS_PREDEFINIDOS[clave]
            self.var_demanda.set(str(d["demanda"]))
            self.var_bermas.set(str(d["bermas"]))
            self.var_sts.set(str(d["sts"]))
            self.var_ocupacion.set(str(d["ocupacion_inicial"]))
            self.var_puertas.set(str(d["puertas"]))
            self.var_impuntuales.set(str(d["impuntuales"]))
            self.var_atraso.set(str(d["atraso_extra_h"]))
            self.var_permanencia.set(str(d["permanencia_h"]))
            self.var_penalizacion.set(str(d["penalizacion"]))

    def abrir_ventana_animacion(self):
        try:
            semilla = (
                None
                if self.var_semilla_libre.get()
                else int(self.var_semilla.get().strip())
            )
            config = {
                "nombre": self.combo_escenario.get(),
                "demanda": float(self.var_demanda.get()),
                "bermas": int(self.var_bermas.get()),
                "sts": int(self.var_sts.get()),
                "ocupacion_inicial": int(self.var_ocupacion.get()),
                "puertas": int(self.var_puertas.get()),
                "impuntuales": float(self.var_impuntuales.get()),
                "atraso_extra_h": float(self.var_atraso.get()),
                "permanencia_h": float(self.var_permanencia.get()),
                "penalizacion": float(self.var_penalizacion.get()),
                "semilla": semilla,
            }
        except Exception as e:
            messagebox.showerror("Error", f"Error en los parámetros:\n{str(e)}")
            return

        self.withdraw()
        VentanaAnimacion(self, config)


# ==============================================================================
# MONITOR VISUAL CON ACELERACIÓN POR LOTES (BATCHING)
# ==============================================================================
class VentanaAnimacion(tk.Toplevel):

    def __init__(self, app_principal, config):
        super().__init__(app_principal)
        self.app_principal = app_principal
        self.config = config
        self.title("Simulación en Tiempo Real - Puerto Sur")
        self.geometry("980x680")
        self.configure(bg="#1a2530")
        self.protocol("WM_DELETE_WINDOW", self.cerrar_todo)

        # Variables de control
        self.en_pausa = False
        self.saltar_animacion = False
        
        # Parámetros dinámicos de velocidad
        self.demora_paso = 0.01
        self.pasos_por_ciclo = 1  # Número de eventos a procesar antes de pausar

        self.construir_pantalla()
        self.iniciar_hilo_simulacion()

    def construir_pantalla(self):
        self.lbl_estado = tk.Label(
            self,
            text="ESTADO: OPERATIVO",
            font=("Segoe UI", 16, "bold"),
            bg="#1a2530",
            fg="#22c55e",
        )
        self.lbl_estado.pack(pady=(12, 2))

        # Tarjetas de Bermas
        frame_bermas = tk.Frame(self, bg="#1a2530")
        frame_bermas.pack(pady=15, fill="x", padx=20)

        self.cards_bermas = []
        n_bermas = self.config["bermas"]
        for i in range(n_bermas):
            sub = tk.Frame(
                frame_bermas, bg="#243447", bd=2, relief="groove", width=130
            )
            sub.pack(side="left", expand=True, fill="both", padx=6, ipady=15)

            tk.Label(
                sub,
                text=f"Berma {i+1}",
                font=("Segoe UI", 11, "bold"),
                bg="#243447",
                fg="#38bdf8",
            ).pack(pady=(4, 8))

            lbl_estado_b = tk.Label(
                sub,
                text="Libre",
                font=("Segoe UI", 10),
                bg="#243447",
                fg="#22c55e",
                justify="center",
            )
            lbl_estado_b.pack(expand=True)
            self.cards_bermas.append((sub, lbl_estado_b))

        # Indicadores de Rada y buques atendidos
        frame_medio = tk.Frame(self, bg="#1a2530")
        frame_medio.pack(fill="x", padx=40, pady=10)

        self.lbl_rada = tk.Label(
            frame_medio,
            text="Buques en Rada (espera): 0",
            font=("Segoe UI", 12, "bold"),
            bg="#1a2530",
            fg="#f59e0b",
        )
        self.lbl_rada.pack(side="left", padx=10)

        self.lbl_finalizados = tk.Label(
            frame_medio,
            text="Buques Atendidos: 0",
            font=("Segoe UI", 12, "bold"),
            bg="#1a2530",
            fg="#22c55e",
        )
        self.lbl_finalizados.pack(side="right", padx=10)

        # Ocupación de Patio
        frame_patio = tk.Frame(self, bg="#1a2530")
        frame_patio.pack(fill="x", padx=40, pady=10)

        self.lbl_patio = tk.Label(
            frame_patio,
            text="Ocupación de Patio: 0 TEU (0.0%)",
            font=("Segoe UI", 10, "bold"),
            bg="#1a2530",
            fg="#94a3b8",
        )
        self.lbl_patio.pack(anchor="w")

        self.barra_patio = ttk.Progressbar(
            frame_patio, maximum=100, length=900, mode="determinate"
        )
        self.barra_patio.pack(fill="x", pady=(4, 0))

        # Reloj de simulación
        self.lbl_reloj = tk.Label(
            self,
            text="MODO ANIMACIÓN VISUAL\nDía 0 - 00:00 hs",
            font=("Segoe UI", 13, "bold"),
            bg="#1a2530",
            fg="#e2e8f0",
        )
        self.lbl_reloj.pack(pady=15)

        # Controles
        frame_controles = tk.Frame(self, bg="#1a2530")
        frame_controles.pack(side="bottom", fill="x", pady=20)

        self.btn_pausa = tk.Button(
            frame_controles,
            text="⏸ PAUSAR",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="white",
            relief="flat",
            command=self.alternar_pausa,
            cursor="hand2",
        )
        self.btn_pausa.pack(side="left", padx=(40, 10), ipady=5, ipadx=10)

        btn_saltear = tk.Button(
            frame_controles,
            text="⏩ SALTEAR ANIMACIÓN",
            font=("Segoe UI", 10, "bold"),
            bg="#ef4444",
            fg="white",
            relief="flat",
            command=self.activar_salto,
            cursor="hand2",
        )
        btn_saltear.pack(side="left", padx=10, ipady=5, ipadx=10)

        # Slider de velocidad expandido a 200
        frame_slider = tk.Frame(frame_controles, bg="#1a2530")
        frame_slider.pack(side="right", padx=(0, 40))

        self.lbl_velocidad_txt = tk.Label(
            frame_slider,
            text="Velocidad Gráfica: 100x",
            font=("Segoe UI", 9, "bold"),
            bg="#1a2530",
            fg="#cbd5e1",
        )
        self.lbl_velocidad_txt.pack()

        self.slider_velocidad = tk.Scale(
            frame_slider,
            from_=1,
            to=200,
            orient="horizontal",
            bg="#1a2530",
            fg="white",
            troughcolor="#334155",
            highlightthickness=0,
            command=self.cambiar_velocidad,
        )
        self.slider_velocidad.set(100)
        self.slider_velocidad.pack()
        self.cambiar_velocidad(100)

    # Actualización visual segura en hilo de Tkinter
    def actualizar_berma(self, idx, texto, color):
        if not self.saltar_animacion and self.pasos_por_ciclo < 25:
            self.after(0, self._render_berma, idx, texto, color)

    def _render_berma(self, idx, texto, color):
        if self.winfo_exists() and idx < len(self.cards_bermas):
            _, lbl = self.cards_bermas[idx]
            lbl.config(text=texto, fg=color)

    def actualizar_rada(self, cantidad):
        if not self.saltar_animacion:
            self.after(
                0,
                lambda: self.lbl_rada.config(
                    text=f"Buques en Rada (espera): {cantidad}"
                ),
            )

    def notificar_patio(self, ocupacion, capacidad):
        if not self.saltar_animacion and self.pasos_por_ciclo < 25:
            pct = 100.0 * ocupacion / capacidad
            self.after(
                0,
                lambda: (
                    self.lbl_patio.config(
                        text=f"Ocupación de Patio: {int(ocupacion)} TEU ({pct:.1f}%)"
                    ),
                    self.barra_patio.config(value=pct),
                ),
            )

    def notificar_buque_finalizado(self, atendidos):
        if not self.saltar_animacion:
            self.after(
                0,
                lambda: self.lbl_finalizados.config(
                    text=f"Buques Atendidos: {atendidos}"
                ),
            )

    def actualizar_reloj(self, hora_sim):
        if not self.saltar_animacion:
            dia = int(hora_sim // 24) + 1
            hora = int(hora_sim % 24)
            minuto = int((hora_sim * 60) % 60)
            texto = f"MODO ANIMACIÓN VISUAL\nDía {dia} - {hora:02d}:{minuto:02d} hs"
            self.after(0, lambda: self.lbl_reloj.config(text=texto))

    def alternar_pausa(self):
        self.en_pausa = not self.en_pausa
        if self.en_pausa:
            self.btn_pausa.config(text="▶ REANUDAR", bg="#16a34a")
            self.lbl_estado.config(text="ESTADO: EN PAUSA", fg="#f59e0b")
        else:
            self.btn_pausa.config(text="⏸ PAUSAR", bg="#0284c7")
            self.lbl_estado.config(text="ESTADO: OPERATIVO", fg="#22c55e")

    def activar_salto(self):
        self.saltar_animacion = True
        self.lbl_estado.config(text="ESTADO: FINALIZANDO...", fg="#ef4444")

    def cambiar_velocidad(self, val):
        """Ajusta la velocidad usando pausas escalonadas y procesamiento por lotes (batching)."""
        v = float(val)
        if v <= 50:
            # Velocidad lenta (detalle paso a paso)
            self.demora_paso = (51.0 - v) * 0.003
            self.pasos_por_ciclo = 1
            texto = f"Velocidad: Lenta ({int(v)})"
        elif v <= 120:
            # Velocidad moderada a rápida
            self.demora_paso = max(0.0005, (120.0 - v) * 0.0002)
            self.pasos_por_ciclo = int(1 + (v - 50) // 10)
            texto = f"Velocidad: Rápida ({int(v)})"
        else:
            # Velocidad Ultra Rápida: ejecución en lotes sin demoras del SO
            self.demora_paso = 0.0001
            self.pasos_por_ciclo = int(10 + (v - 120) * 1.5)  # hasta ~130 eventos por ciclo
            texto = f"Velocidad: ULTRA RÁPIDA ({int(v)})"

        if hasattr(self, "lbl_velocidad_txt"):
            self.lbl_velocidad_txt.config(text=texto)

    def iniciar_hilo_simulacion(self):
        hilo = threading.Thread(target=self._correr_bucle_simpy, daemon=True)
        hilo.start()

    def _correr_bucle_simpy(self):
        datos, origen = cargar_datos()
        rng = np.random.default_rng(self.config["semilla"])

        df_escalas = datos.copy()
        df_escalas["hora_sim"] /= float(self.config["demanda"])

        env = simpy.Environment()
        puerto = PuertoVisual(env, origen, self.config, rng, self)

        env.process(generador_llegadas(env, puerto, df_escalas))
        horizonte = float(df_escalas["hora_sim"].max()) + 14 * 24

        # Bucle con procesamiento optimizado por lotes
        while env.peek() < horizonte:
            if self.en_pausa:
                time.sleep(0.1)
                continue

            if self.saltar_animacion:
                # Si el usuario saltó la animación, avanza sin pausas
                env.step()
                continue

            # Procesar un bloque de eventos simultáneos según la velocidad fijada
            for _ in range(self.pasos_por_ciclo):
                if env.peek() >= horizonte:
                    break
                env.step()

            self.actualizar_reloj(env.now)
            if self.demora_paso > 0:
                time.sleep(self.demora_paso)

        puerto.actualizar_patio()
        r = pd.DataFrame(puerto.registros)
        fin = len(r)

        resumen = {
            "escenario": self.config.get("nombre", "Personalizado"),
            "total_buques": len(df_escalas),
            "atendidos": fin,
            "pendientes": len(df_escalas) - fin,
            "estadia_media": (
                float(r["estadia_total_h"].mean()) if fin else 0.0
            ),
            "espera_media_rada": (
                float(r["espera_rada_h"].mean()) if fin else 0.0
            ),
            "cumple_16h_pct": (
                100.0 * float((r["estadia_total_h"] <= 16).mean())
                if fin else 0.0
            ),
            "ocupacion_max_patio_pct": 100.0
            * puerto.ocupacion_maxima
            / CAPACIDAD_PATIO,
            "tiempo_sobre_75_pct": 100.0
            * puerto.tiempo_congestion
            / max(env.now, 1),
            "util_bermas_pct": 100.0
            * puerto.uso_bermas
            / (self.config["bermas"] * env.now),
            "util_sts_pct": 100.0
            * puerto.uso_gruas
            / (self.config["sts"] * env.now),
            "horizonte_h": env.now,
        }

        self.after(0, self.mostrar_resultados_finales, resumen)

    def mostrar_resultados_finales(self, r):
        self.destroy()
        self.app_principal.deiconify()

        v = tk.Toplevel(self.app_principal)
        v.title("Resultados de la Simulación")
        v.geometry("500x460")
        v.configure(bg="#1e293b")

        tk.Label(
            v,
            text=f"📊 RESULTADOS: {r['escenario'].upper()}",
            font=("Segoe UI", 12, "bold"),
            bg="#1e293b",
            fg="#38bdf8",
        ).pack(pady=12)

        lineas = [
            f"Buques arribados: {r['total_buques']}",
            f"Buques atendidos: {r['atendidos']} (Pendientes: {r['pendientes']})",
            f"Espera media en rada: {r['espera_media_rada']:.2f} h",
            f"Estadía media: {r['estadia_media']:.2f} h",
            f"Cumplimiento estadía <= 16 h: {r['cumple_16h_pct']:.2f}%",
            f"Ocupación máxima patio: {r['ocupacion_max_patio_pct']:.2f}%",
            f"Tiempo congestión patio (>75%): {r['tiempo_sobre_75_pct']:.2f}%",
            f"Utilización Bermas: {r['util_bermas_pct']:.2f}%",
            f"Utilización STS: {r['util_sts_pct']:.2f}%",
            f"Horizonte total simulado: {r['horizonte_h']:.1f} h",
        ]

        frame_metricas = tk.Frame(v, bg="#0f172a", bd=1, relief="solid")
        frame_metricas.pack(padx=20, pady=10, fill="both", expand=True)

        for l in lineas:
            tk.Label(
                frame_metricas,
                text=l,
                bg="#0f172a",
                fg="white",
                font=("Consolas", 10),
                anchor="w",
            ).pack(fill="x", padx=15, pady=4)

        tk.Button(
            v,
            text="Cerrar",
            bg="#475569",
            fg="white",
            relief="flat",
            command=v.destroy,
        ).pack(pady=10)

    def cerrar_todo(self):
        self.destroy()
        self.app_principal.destroy()


# ==============================================================================
# ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    app = AppSimulador()
    app.mainloop()
    
    
    

