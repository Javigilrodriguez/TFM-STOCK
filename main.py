import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import Calendar
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

from modelo import ModeloPrediccion
from optimizador import Optimizador
from utilidades import Utilidades
from calendario import CalendarioProduccion

class SistemaStock:
    def __init__(self, root):
        """Inicializa la aplicación"""
        self.root = root
        self.root.title("Sistema de Planificación de Producción")
        self.root.geometry("800x800")
        
        # Variables de control
        self.ruta_carpeta = tk.StringVar()
        self.modelo_cargado = tk.StringVar(value="Estado: No hay modelo cargado")
        self.modelo_entrenado = False
        
        # Calendario y fechas
        self.calendario = CalendarioProduccion()
        self.fecha_inicio = tk.StringVar()
        self.fecha_fin = tk.StringVar()
        
        # Inicializar con fechas por defecto
        fecha_actual = datetime.now()
        self.fecha_inicio.set(fecha_actual.strftime('%Y-%m-%d'))
        self.fecha_fin.set((fecha_actual + timedelta(days=7)).strftime('%Y-%m-%d'))
        
        # Parámetros de producción
        self.params = {
            'dias_stock_seguridad': tk.StringVar(value='3'),
            'horas_min_produccion': tk.StringVar(value='2'),
            'horas_disponibles': tk.StringVar(value='100'),
            'horas_mantenimiento': tk.StringVar(value='5')
        }
        
        # Inicializar componentes
        self.utilidades = Utilidades()
        self.modelo = ModeloPrediccion()
        self.optimizador = None
        
        # Verificar si hay modelo guardado y actualizar estado
        modelos = self.modelo.listar_modelos()
        if modelos:
            modelo_actual = modelos[0]
            self.modelo_cargado.set(
                f"Modelo cargado: {modelo_actual['nombre']} "
                f"(Fecha: {modelo_actual['fecha']})"
            )
            self.modelo_entrenado = True
        
        # Crear interfaz
        self.crear_widgets()
    
    def crear_widgets(self):
        """Crea la interfaz gráfica"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Estado del modelo
        status_frame = ttk.LabelFrame(main_frame, text="Estado del Modelo", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, textvariable=self.modelo_cargado).pack()
        
        # Sección de selección de carpeta
        folder_frame = ttk.LabelFrame(main_frame, text="Selección de Datos", padding=10)
        folder_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(folder_frame, textvariable=self.ruta_carpeta, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="Seleccionar Carpeta", 
                  command=self.seleccionar_carpeta).pack(side=tk.LEFT, padx=5)
        
        # Calendario y Fechas
        calendar_frame = ttk.LabelFrame(main_frame, text="Calendario de Producción", padding=10)
        calendar_frame.pack(fill=tk.X, pady=5)
        
        # Frame para fechas
        dates_frame = ttk.Frame(calendar_frame)
        dates_frame.pack(fill=tk.X, pady=5)
        
        # Fecha inicio
        ttk.Label(dates_frame, text="Fecha Inicio:").pack(side=tk.LEFT, padx=5)
        self.fecha_inicio_entry = ttk.Entry(dates_frame, textvariable=self.fecha_inicio, width=10)
        self.fecha_inicio_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dates_frame, text="...", 
                  command=lambda: self.seleccionar_fecha('inicio')).pack(side=tk.LEFT)
        
        # Fecha fin
        ttk.Label(dates_frame, text="Fecha Fin:").pack(side=tk.LEFT, padx=5)
        self.fecha_fin_entry = ttk.Entry(dates_frame, textvariable=self.fecha_fin, width=10)
        self.fecha_fin_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(dates_frame, text="...", 
                  command=lambda: self.seleccionar_fecha('fin')).pack(side=tk.LEFT)
        
        # Calendario
        self.calendar = Calendar(calendar_frame, selectmode='day', date_pattern='y-mm-dd')
        self.calendar.pack(pady=5)
        
        # Configurar estilo del calendario
        self.calendar.tag_config('working_day', background='lightgreen')
        
        # Botones de calendario
        cal_buttons_frame = ttk.Frame(calendar_frame)
        cal_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(cal_buttons_frame, text="Marcar como Hábil", 
                  command=lambda: self.marcar_dia(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(cal_buttons_frame, text="Marcar como No Hábil", 
                  command=lambda: self.marcar_dia(False)).pack(side=tk.LEFT, padx=5)
        
        # Botón de entrenamiento
        train_frame = ttk.LabelFrame(main_frame, text="Entrenamiento", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        ttk.Button(train_frame, text="Entrenar Modelo", 
                  command=self.entrenar_modelo).pack(pady=5)
        
        # Sección de parámetros
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Producción", padding=10)
        params_frame.pack(fill=tk.X, pady=5)
        
        for key, label in {
            'dias_stock_seguridad': 'Días de Stock Seguridad',
            'horas_min_produccion': 'Horas Mínimas Producción',
            'horas_disponibles': 'Horas Disponibles',
            'horas_mantenimiento': 'Horas Mantenimiento'
        }.items():
            frame = ttk.Frame(params_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=self.params[key], width=10).pack(side=tk.RIGHT)
        
        # Botón de optimización
        optim_frame = ttk.LabelFrame(main_frame, text="Optimización", padding=10)
        optim_frame.pack(fill=tk.X, pady=5)
        ttk.Button(optim_frame, text="Optimizar Producción", 
                  command=self.optimizar_produccion).pack(pady=5)
        
        # Cargar estado inicial del calendario
        self.cargar_estado_calendario()
    
    def cargar_estado_calendario(self):
        """Carga el estado guardado del calendario en la interfaz"""
        # Limpiar todos los eventos existentes
        self.calendar.calevent_remove("all")
        
        # Configurar el estilo para días hábiles
        self.calendar.tag_config('working_day', background='lightgreen')
        
        # Cargar todos los días hábiles
        for fecha_str, es_habil in self.calendario.dias_habiles.items():
            if es_habil:
                try:
                    fecha = datetime.strptime(fecha_str, '%Y-%m-%d')
                    self.calendar.calevent_create(fecha, 'Día Hábil', 'working_day')
                except Exception as e:
                    print(f"Error cargando fecha {fecha_str}: {e}")
                    continue
    
    def marcar_dia(self, es_habil):
        """Marca o desmarca un día como hábil"""
        fecha = datetime.strptime(self.calendar.get_date(), '%Y-%m-%d')
        fecha_str = fecha.strftime('%Y-%m-%d')
        self.calendario.establecer_dia_habil(fecha, es_habil)
        
        # Limpiar eventos solo para esta fecha específica
        self.calendar.calevent_remove("all", fecha)
        
        # Si es hábil, crear evento nuevo
        if es_habil:
            self.calendar.calevent_create(fecha, 'Día Hábil', 'working_day')
        
        # Cargar todos los días hábiles
        self.cargar_estado_calendario()
    
    def seleccionar_fecha(self, tipo):
        """Abre un selector de fecha"""
        def set_date():
            if tipo == 'inicio':
                self.fecha_inicio.set(cal.get_date())
            else:
                self.fecha_fin.set(cal.get_date())
            top.destroy()
        
        top = tk.Toplevel(self.root)
        cal = Calendar(top, selectmode='day', date_pattern='y-mm-dd')
        cal.pack(padx=10, pady=10)
        ttk.Button(top, text="Seleccionar", command=set_date).pack(pady=5)
    
    def seleccionar_carpeta(self):
        """Permite al usuario seleccionar una carpeta con datasets"""
        carpeta = filedialog.askdirectory(title="Seleccionar Carpeta con Datasets")
        if carpeta:
            self.ruta_carpeta.set(carpeta)
            messagebox.showinfo("Info", "Carpeta seleccionada. Puede proceder con el entrenamiento.")
    
    def cargar_datasets(self):
        """Carga todos los datasets de la carpeta seleccionada"""
        try:
            carpeta = self.ruta_carpeta.get()
            if not carpeta:
                raise ValueError("Debe seleccionar una carpeta primero")
            
            datasets = []
            archivos_validos = [
                f for f in os.listdir(carpeta) 
                if f.endswith('.csv') and f.startswith('Dataset')
            ]
            
            print(f"Archivos encontrados: {archivos_validos}")
            
            for archivo in archivos_validos:
                ruta_completa = os.path.join(carpeta, archivo)
                try:
                    # Usar el método de la clase Utilidades
                    df = self.utilidades.leer_csv(ruta_completa)
                    
                    if not df.empty:
                        datasets.append(df)
                        print(f"Archivo cargado: {archivo}")
                
                except Exception as e:
                    print(f"Error cargando {archivo}: {e}")
                    continue
            
            if not datasets:
                raise ValueError("No se encontraron datos válidos en los archivos")
            
            # Combinar todos los DataFrames
            df_combinado = pd.concat(datasets, ignore_index=True)
            
            print(f"\nColumnas encontradas:")
            print(df_combinado.columns.tolist())
            print(f"Total de registros: {len(df_combinado)}")
            
            return df_combinado

        except Exception as e:
            print(f"Error en cargar_datasets: {e}")
            raise

    def entrenar_modelo(self):
        """Entrena el modelo con los datos seleccionados."""
        try:
            if not self.ruta_carpeta.get():
                messagebox.showerror("Error", "Seleccione una carpeta primero")
                return

            # Cargar y procesar datos
            df_combinado = self.cargar_datasets()
            df_procesado = self.utilidades.procesar_datos(df_combinado)

            if not self.utilidades.validar_datos(df_procesado):
                raise ValueError("Los datos no cumplen con los requisitos mínimos")

            # Entrenar modelo
            X = self.modelo.preparar_caracteristicas(df_procesado)
            y = df_procesado['M_VTA -15']
            self.modelo.entrenar(X, y)

            # Actualizar estado
            modelos = self.modelo.listar_modelos()
            if modelos:
                modelo_actual = modelos[0]
                self.modelo_cargado.set(
                    f"Modelo cargado: {modelo_actual['nombre']} "
                    f"(Fecha: {modelo_actual['fecha']})"
                )

            self.modelo_entrenado = True
            messagebox.showinfo("Éxito", "Modelo entrenado y guardado correctamente")

        except Exception as e:
            messagebox.showerror("Error", f"Error en entrenamiento: {str(e)}")

    def optimizar_produccion(self):
        """Realiza la optimización de la producción"""
        if not self.modelo_entrenado:
            messagebox.showerror("Error", "No hay modelo válido cargado")
            return
            
        try:
            # Validar fechas
            try:
                fecha_inicio = datetime.strptime(self.fecha_inicio.get(), '%Y-%m-%d')
                fecha_fin = datetime.strptime(self.fecha_fin.get(), '%Y-%m-%d')
                
                if fecha_inicio > fecha_fin:
                    raise ValueError("La fecha de inicio debe ser anterior a la fecha fin")
                
                dias_habiles = self.calendario.obtener_dias_habiles_rango(fecha_inicio, fecha_fin)
                if not dias_habiles:
                    raise ValueError("No hay días hábiles en el rango seleccionado")
                
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return
            
            # Cargar datos actuales
            df_actual = self.cargar_datasets()
            df_procesado = self.utilidades.procesar_datos(df_actual)
            
            # Agregar información de días hábiles
            df_procesado['dias_habiles'] = len(dias_habiles)
            
            # Realizar predicción
            X = self.modelo.preparar_caracteristicas(df_procesado)
            df_procesado['DEMANDA_PREDICHA'] = self.modelo.predecir(X)
            
            # Inicializar optimizador si no existe
            if not self.optimizador:
                self.optimizador = Optimizador(
                    dias_stock_seguridad=int(self.params['dias_stock_seguridad'].get()),
                    horas_min_produccion=float(self.params['horas_min_produccion'].get())
                )
            
            # Optimizar producción
            horas_produccion, cantidades_produccion, _ = self.optimizador.optimizar_produccion(
                df_procesado,
                float(self.params['horas_disponibles'].get()),
                float(self.params['horas_mantenimiento'].get())
            )
            
            # Preparar resultados
            df_procesado['CAJAS_PRODUCIR'] = cantidades_produccion
            df_procesado['HORAS_PRODUCCION'] = horas_produccion
            df_procesado['FECHA_INICIO'] = self.fecha_inicio.get()
            df_procesado['FECHA_FIN'] = self.fecha_fin.get()
            df_procesado['DIAS_HABILES'] = len(dias_habiles)
            
            # Guardar resultados
            ruta_salida = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="plan_produccion.csv"
            )
            
            if ruta_salida:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                
                # Guardar archivo
                df_procesado.to_csv(ruta_salida, index=False, sep=';')
                
                # Mostrar resumen
                total_cajas = df_procesado['CAJAS_PRODUCIR'].sum()
                total_horas = df_procesado['HORAS_PRODUCCION'].sum()
                productos_programados = (df_procesado['CAJAS_PRODUCIR'] > 0).sum()
                
                mensaje = (
                    f"Plan de producción generado exitosamente\n\n"
                    f"Período: {self.fecha_inicio.get()} a {self.fecha_fin.get()}\n"
                    f"Días hábiles: {len(dias_habiles)}\n"
                    f"Productos programados: {productos_programados}\n"
                    f"Total cajas: {total_cajas:.0f}\n"
                    f"Total horas: {total_horas:.1f}\n\n"
                    f"Plan guardado en: {Path(ruta_salida).name}"
                )
                
                messagebox.showinfo("Éxito", mensaje)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en optimización: {str(e)}")
            print(f"Error detallado: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    try:
        root = tk.Tk()
        app = SistemaStock(root)
        root.mainloop()
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()