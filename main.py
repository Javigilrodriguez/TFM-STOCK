import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio
import plotly.graph_objects as go
from PIL import Image, ImageTk
import io
from typing import Optional
import os

from modelo import ModeloPrediccion
from optimizador import Optimizador
from utilidades import Utilidades

class SistemaStock:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Planificación de Producción")
        self.root.geometry("1024x768")
        
        # Configurar canvas scrollable
        self.main_canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        # Configurar scroll
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind mousewheel
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        # Variables de control
        self.ruta_carpeta = tk.StringVar()
        self.modelo_cargado = tk.StringVar(value="Estado: No hay modelo cargado")
        self.usar_prediccion = tk.BooleanVar(value=False)
        
        # Parámetros
        self.params = {
            'dias_stock_seguridad': tk.StringVar(value='3'),
            'horas_min_produccion': tk.StringVar(value='2'),
            'horas_disponibles': tk.StringVar(value='100'),
            'horas_mantenimiento': tk.StringVar(value='5')
        }
        
        # Componentes
        self.utilidades = Utilidades()
        self.optimizador = None
        self.modelo = None
        
        # Crear interfaz
        self.crear_widgets()
        self.configurar_layout()
        
    def _on_mousewheel(self, event):
        """Maneja el scroll con la rueda del mouse"""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def crear_widgets(self):
        """Crea todos los widgets de la interfaz"""
        # Frame principal con padding
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Estado
        status_frame = ttk.LabelFrame(main_frame, text="Estado del Sistema", padding="5")
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, textvariable=self.modelo_cargado).pack()
        
        # Selección de datos
        data_frame = ttk.LabelFrame(main_frame, text="Datos de Entrada", padding="5")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(data_frame, text="Carpeta de datos:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(data_frame, textvariable=self.ruta_carpeta, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_frame, text="Examinar", command=self.seleccionar_carpeta).pack(side=tk.LEFT)
        
        # Parámetros
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Optimización", padding="5")
        params_frame.pack(fill=tk.X, pady=5)
        
        for key, label in {
            'dias_stock_seguridad': 'Días de stock seguridad:',
            'horas_min_produccion': 'Horas mínimas producción:',
            'horas_disponibles': 'Horas disponibles:',
            'horas_mantenimiento': 'Horas mantenimiento:'
        }.items():
            param_row = ttk.Frame(params_frame)
            param_row.pack(fill=tk.X, pady=2)
            ttk.Label(param_row, text=label, width=30).pack(side=tk.LEFT)
            ttk.Entry(param_row, textvariable=self.params[key], width=10).pack(side=tk.LEFT)
        
        # Frame para selección de período
        period_frame = ttk.LabelFrame(params_frame, text="Período de Planificación", padding="5")
        period_frame.pack(fill=tk.X, pady=5)
        
        fecha_inicio_row = ttk.Frame(period_frame)
        fecha_inicio_row.pack(fill=tk.X, pady=2)
        ttk.Label(fecha_inicio_row, text="Fecha Inicio:", width=30).pack(side=tk.LEFT)
        self.cal_inicio = DateEntry(fecha_inicio_row, width=12, background='darkblue',
                                  foreground='white', borderwidth=2,
                                  date_pattern='yyyy-mm-dd')
        self.cal_inicio.pack(side=tk.LEFT)
        
        fecha_fin_row = ttk.Frame(period_frame)
        fecha_fin_row.pack(fill=tk.X, pady=2)
        ttk.Label(fecha_fin_row, text="Fecha Fin:", width=30).pack(side=tk.LEFT)
        self.cal_fin = DateEntry(fecha_fin_row, width=12, background='darkblue',
                               foreground='white', borderwidth=2,
                               date_pattern='yyyy-mm-dd')
        self.cal_fin.pack(side=tk.LEFT)
        
        # Predicción
        ttk.Checkbutton(
            params_frame,
            text="Usar predicción de demanda",
            variable=self.usar_prediccion
        ).pack(pady=5)
        
        # Acciones
        actions_frame = ttk.LabelFrame(main_frame, text="Acciones", padding="5")
        actions_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(actions_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Cargar Datos", 
                  command=self.cargar_datos).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Entrenar Modelo", 
                  command=self.entrenar_modelo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Generar Planificación", 
                  command=self.generar_planificacion).pack(side=tk.LEFT, padx=5)
        
        # Visualización
        viz_frame = ttk.LabelFrame(main_frame, text="Visualización", padding="5")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        viz_buttons = ttk.Frame(viz_frame)
        viz_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(viz_buttons, text="Ver Evolución", 
                  command=self.mostrar_evolucion).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_buttons, text="Comparar con Plan Propuesto", 
                  command=self.mostrar_comparacion).pack(side=tk.LEFT, padx=5)
        
        # Canvas para gráficos
        self.grafico_canvas = tk.Canvas(viz_frame, height=400)
        self.grafico_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def configurar_layout(self):
        """Configura el layout principal con scroll"""
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def seleccionar_carpeta(self):
        """Permite seleccionar la carpeta de datos"""
        carpeta = filedialog.askdirectory(title="Seleccionar Carpeta con Datasets")
        if carpeta:
            self.ruta_carpeta.set(carpeta)
            messagebox.showinfo("Info", "Carpeta seleccionada correctamente")
            
    def cargar_datos(self):
        """Carga los datasets de la carpeta seleccionada"""
        try:
            carpeta = self.ruta_carpeta.get()
            if not carpeta:
                messagebox.showwarning("Advertencia", 
                    "Debe seleccionar una carpeta primero")
                return
                
            self.df_actual = self.utilidades.cargar_datasets(carpeta)
            
            if self.df_actual is not None and not self.df_actual.empty:
                messagebox.showinfo("Éxito", 
                    f"Datos cargados correctamente\n"
                    f"Registros: {len(self.df_actual)}")
            else:
                messagebox.showerror("Error",
                    "No se pudieron cargar los datos correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando datos: {str(e)}")
            
    def entrenar_modelo(self):
        """Entrena el modelo predictivo si está activado"""
        try:
            if not hasattr(self, 'df_actual'):
                messagebox.showerror("Error", "Debe cargar datos primero")
                return
                
            if not self.usar_prediccion.get():
                messagebox.showinfo("Info", "La predicción está desactivada")
                return
                
            if not self.modelo:
                self.modelo = ModeloPrediccion()
                
            X = self.modelo.preparar_caracteristicas(self.df_actual)
            y = self.df_actual['M_Vta -15'].astype(float)
            
            self.modelo.entrenar(X, y)
            
            self.modelo_cargado.set("Estado: Modelo entrenado correctamente")
            messagebox.showinfo("Éxito", "Modelo entrenado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en entrenamiento: {str(e)}")
            
    def generar_planificacion(self):
        """Ejecuta la optimización de producción para el período seleccionado"""
        try:
            if not hasattr(self, 'df_actual'):
                raise ValueError("Debe cargar datos primero")
                
            # Obtener fechas de los calendarios
            fecha_inicio = self.cal_inicio.get_date()
            fecha_fin = self.cal_fin.get_date()
            
            # Validar fechas
            if fecha_fin < fecha_inicio:
                messagebox.showerror("Error", "La fecha fin debe ser posterior a la fecha inicio")
                return
                
            # Calcular días del período
            dias_periodo = (fecha_fin - fecha_inicio).days + 1
                
            # Crear optimizador si no existe
            if not self.optimizador:
                self.optimizador = Optimizador(
                    dias_stock_seguridad=int(self.params['dias_stock_seguridad'].get()),
                    horas_min_produccion=float(self.params['horas_min_produccion'].get())
                )
            
            # Crear DataFrame temporal con las columnas necesarias
            df_temp = self.df_actual.copy()
            
            # Asegurar que existan todas las columnas requeridas
            columnas_requeridas = ['COD_ART', 'NOM_ART', 'Disponible', 'Cj/H', 'DEMANDA_PREDICHA']
            for col in columnas_requeridas:
                if col not in df_temp.columns:
                    df_temp[col] = 0
            
            # Optimizar considerando todo el período
            horas_total = float(self.params['horas_disponibles'].get()) * dias_periodo
            mant_total = float(self.params['horas_mantenimiento'].get()) * dias_periodo
            
            horas, cantidades, objetivo = self.optimizador.optimizar_produccion(
                df_temp,
                horas_total,
                mant_total
            )
            
            # Guardar resultados
            self.df_actual['HORAS_PRODUCCION'] = horas
            self.df_actual['CANTIDADES_PRODUCCION'] = cantidades
            
            # Calcular métricas por período
            self.df_actual['PRODUCCION_DIARIA'] = cantidades / dias_periodo
            self.df_actual['HORAS_DIARIAS'] = horas / dias_periodo
            
            # Mostrar resultados
            messagebox.showinfo(
                "Éxito", 
                f"Optimización completada para {dias_periodo} días:\n"
                f"Total horas: {horas.sum():.1f}\n"
                f"Total cajas: {cantidades.sum():.0f}\n"
                f"Productos programados: {(cantidades > 0).sum()}"
            )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en optimización: {str(e)}")
            raise
            
    def mostrar_evolucion(self):
        """Muestra gráfico de evolución de métricas"""
        try:
            if not hasattr(self, 'df_actual') or 'CANTIDADES_PRODUCCION' not in self.df_actual.columns:
                raise ValueError("No hay datos de optimización")
                
            # Crear gráfico de evolución
            fig = go.Figure()
            
            # Agregar línea de producción
            fig.add_trace(go.Scatter(
                x=self.df_actual['COD_ART'],
                y=self.df_actual['CANTIDADES_PRODUCCION'],
                name='Producción Planificada',
                mode='lines+markers'
            ))
            
            # Agregar línea de stock
            fig.add_trace(go.Scatter(
                x=self.df_actual['COD_ART'],
                y=self.df_actual['Disponible'],
                name='Stock Actual',
                mode='lines+markers'
            ))
            
            # Configurar layout
            fig.update_layout(
                title='Evolución de Stock y Producción',
                xaxis_title='Código de Artículo',
                yaxis_title='Cantidad (cajas)',
                height=400
            )
            
            # Convertir a imagen
            img_bytes = pio.to_image(fig, format="png")
            img = Image.open(io.BytesIO(img_bytes))
            
            # Ajustar tamaño
            width = self.grafico_canvas.winfo_width()
            height = 400
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Mostrar en canvas
            self.grafico_photo = ImageTk.PhotoImage(img)
            self.grafico_canvas.delete("all")
            self.grafico_canvas.create_image(0, 0, anchor="nw", image=self.grafico_photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando evolución: {str(e)}")
            
    def mostrar_comparacion(self):
        """Muestra comparación con datos reales del archivo de planificación"""
        try:
            if not hasattr(self, 'df_actual'):
                raise ValueError("No hay datos de optimización")
                
            # Permitir seleccionar archivo de planificación propuesta
            archivo_plan = filedialog.askopenfilename(
                title="Seleccionar archivo de planificación propuesta",
                filetypes=[("Excel files", "*.xlsx")]
            )
            
            if archivo_plan:
                # Leer archivo Excel
                df_plan = pd.read_excel(archivo_plan)
                
                # Asegurar que tenemos las columnas necesarias
                cols_req = ['COD_ART', 'CANTIDADES_PRODUCCION']
                if not all(col in df_plan.columns for col in cols_req):
                    raise ValueError(
                        "El archivo debe contener las columnas: " +
                        ", ".join(cols_req)
                    )
                
                # Realizar comparación
                df_comp = pd.merge(
                    self.df_actual[['COD_ART', 'CANTIDADES_PRODUCCION']], 
                    df_plan[cols_req],
                    on='COD_ART',
                    suffixes=('_calc', '_real')
                )
                
                # Calcular diferencias
                df_comp['diferencia'] = (
                    df_comp['CANTIDADES_PRODUCCION_calc'] - 
                    df_comp['CANTIDADES_PRODUCCION_real']
                )
                
                df_comp['diferencia_porcentual'] = (
                    df_comp['diferencia'] / 
                    df_comp['CANTIDADES_PRODUCCION_real'] * 100
                ).round(2)
                
                # Mostrar métricas
                error_medio = abs(df_comp['diferencia']).mean()
                error_porc = abs(df_comp['diferencia_porcentual']).mean()
                productos_diff = (abs(df_comp['diferencia']) > 0).sum()
                
                messagebox.showinfo(
                    "Comparación con Planificación Propuesta",
                    f"Diferencia media: {error_medio:.2f} unidades\n"
                    f"Error porcentual medio: {error_porc:.1f}%\n"
                    f"Productos con diferencias: {productos_diff}"
                )
                
                # Guardar comparación
                ruta_comp = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    initialfile="comparacion_planificacion.csv"
                )
                
                if ruta_comp:
                    df_comp.to_csv(ruta_comp, index=False, sep=';')
                    
                # Generar gráfico de comparación
                fig = go.Figure()
                
                # Agregar barras de producción calculada
                fig.add_trace(go.Bar(
                    x=df_comp['COD_ART'],
                    y=df_comp['CANTIDADES_PRODUCCION_calc'],
                    name='Planificación Calculada'
                ))
                
                # Agregar barras de producción real
                fig.add_trace(go.Bar(
                    x=df_comp['COD_ART'],
                    y=df_comp['CANTIDADES_PRODUCCION_real'],
                    name='Planificación Propuesta'
                ))
                
                # Configurar layout
                fig.update_layout(
                    title='Comparación de Planificaciones',
                    xaxis_title='Código de Artículo',
                    yaxis_title='Cantidad (cajas)',
                    barmode='group',
                    height=400
                )
                
                # Mostrar gráfico
                img_bytes = pio.to_image(fig, format="png")
                img = Image.open(io.BytesIO(img_bytes))
                
                # Ajustar tamaño
                width = self.grafico_canvas.winfo_width()
                height = 400
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Mostrar en canvas
                self.grafico_photo = ImageTk.PhotoImage(img)
                self.grafico_canvas.delete("all")
                self.grafico_canvas.create_image(0, 0, anchor="nw", image=self.grafico_photo)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")

def main():
    root = tk.Tk()
    app = SistemaStock(root)
    root.mainloop()

if __name__ == "__main__":
    main()