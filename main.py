import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio
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
        ttk.Button(viz_buttons, text="Comparar Stock", 
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
            # Verificar si hay carpeta seleccionada
            carpeta = self.ruta_carpeta.get()
            if not carpeta:
                messagebox.showwarning("Advertencia", 
                    "Debe seleccionar una carpeta primero.\n\n"
                    "Use el botón 'Examinar' para elegir la carpeta\n"
                    "que contiene los archivos CSV")
                return
            
            # Verificar si la carpeta existe
            if not os.path.exists(carpeta):
                messagebox.showerror("Error",
                    f"La carpeta seleccionada no existe:\n{carpeta}\n\n"
                    "Por favor, seleccione una carpeta válida.")
                return
                
            # Buscar archivos CSV en la carpeta
            archivos_csv = [f for f in os.listdir(carpeta) 
                        if f.lower().endswith('.csv')]
            
            if not archivos_csv:
                messagebox.showerror("Error",
                    "No se encontraron archivos CSV en la carpeta.\n\n"
                    "Asegúrese de que la carpeta contiene archivos .csv")
                return
            
            # Usar utilidades para cargar datos
            self.df_actual = self.utilidades.cargar_datasets(carpeta)
            
            if self.df_actual is not None and not self.df_actual.empty:
                messagebox.showinfo("Éxito", 
                    f"Datos cargados correctamente\n\n"
                    f"Archivos procesados: {len(archivos_csv)}\n"
                    f"Registros totales: {len(self.df_actual)}\n\n"
                    f"Puede proceder a generar la planificación")
            else:
                messagebox.showerror("Error",
                    "No se pudieron cargar los datos correctamente.\n"
                    "Verifique que los archivos tienen el formato esperado.")
            
        except Exception as e:
            messagebox.showerror("Error", 
                f"Error cargando datos: {str(e)}\n\n"
                "Verifique que los archivos tienen el formato correcto.")
            
    def entrenar_modelo(self):
        """Entrena el modelo predictivo si está activado"""
        try:
            # Verificar si hay datos cargados
            if not hasattr(self, 'df_actual'):
                messagebox.showerror("Error", "Debe cargar datos primero usando el botón 'Cargar Datos'")
                return
            
            # Verificar si la predicción está activada    
            if not self.usar_prediccion.get():
                messagebox.showinfo("Info", "La predicción de demanda está desactivada. Active la casilla para usar el modelo predictivo.")
                return
                
            # Crear modelo si no existe
            if not self.modelo:
                self.modelo = ModeloPrediccion()
                print("Modelo predictivo inicializado")
            
            # Preparar datos para entrenamiento
            print("Preparando datos para entrenamiento...")
            print(f"Columnas disponibles: {self.df_actual.columns.tolist()}")
            
            # Verificar columnas necesarias
            columnas_req = ['M_Vta -15', 'Vta -15', 'M_Vta -15 AA']
            faltantes = [col for col in columnas_req if col not in self.df_actual.columns]
            if faltantes:
                raise ValueError(f"Faltan columnas requeridas para entrenamiento: {faltantes}")
            
            # Preparar características y entrenar
            X = self.modelo.preparar_caracteristicas(self.df_actual)
            y = self.df_actual['M_Vta -15'].astype(float)
            
            print("Iniciando entrenamiento...")
            print(f"Datos de entrenamiento: {len(X)} registros")
            print(f"Características utilizadas: {X.columns.tolist()}")
            
            self.modelo.entrenar(X, y)
            
            self.modelo_cargado.set("Estado: Modelo entrenado correctamente")
            messagebox.showinfo("Éxito", 
                              "Modelo entrenado correctamente\n\n"
                              f"Registros utilizados: {len(X)}\n"
                              f"Características: {len(X.columns)}")
            
        except Exception as e:
            print(f"Error detallado en entrenamiento: {str(e)}")
            messagebox.showerror("Error", 
                               f"Error en entrenamiento: {str(e)}\n\n"
                               "Revise la consola para más detalles")
            
    def generar_planificacion(self):
        """Ejecuta la optimización de producción"""
        try:
            if not hasattr(self, 'df_actual'):
                raise ValueError("Debe cargar datos primero")
                
            # Crear optimizador si no existe
            if not self.optimizador:
                self.optimizador = Optimizador(
                    dias_stock_seguridad=int(self.params['dias_stock_seguridad'].get()),
                    horas_min_produccion=float(self.params['horas_min_produccion'].get())
                )
            
            # Optimizar
            horas, cantidades, objetivo = self.optimizador.optimizar_produccion(
                self.df_actual,
                float(self.params['horas_disponibles'].get()),
                float(self.params['horas_mantenimiento'].get())
            )
            
            # Guardar resultados
            self.df_actual['HORAS_PRODUCCION'] = horas
            self.df_actual['CANTIDADES_PRODUCCION'] = cantidades
            
            # Guardar archivo
            ruta_salida = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="plan_produccion.csv"
            )
            
            if ruta_salida:
                self.df_actual.to_csv(ruta_salida, index=False, sep=';')
                messagebox.showinfo("Éxito", 
                    f"Optimización completada:\n"
                    f"Total horas: {horas.sum():.1f}\n"
                    f"Total cajas: {cantidades.sum():.0f}\n"
                    f"Productos programados: {(cantidades > 0).sum()}"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en optimización: {str(e)}")
            
    def mostrar_evolucion(self):
        """Muestra gráfico de evolución de métricas"""
        try:
            if not self.optimizador:
                raise ValueError("No hay datos de optimización")
                
            fig = self.optimizador.generar_grafico_evolucion()
            if fig:
                self._mostrar_grafico(fig)
            else:
                messagebox.showinfo("Info", "No hay datos suficientes para mostrar evolución")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error mostrando evolución: {str(e)}")
            
    def mostrar_comparacion(self):
        """Muestra gráfico comparativo de stock"""
        try:
            if not self.optimizador:
                raise ValueError("No hay datos de optimización")
                
            # Permitir seleccionar archivo de datos reales
            archivo_real = filedialog.askopenfilename(
                title="Seleccionar datos reales",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if archivo_real:
                df_real = pd.read_csv(archivo_real, sep=';')
                fig = self.optimizador.generar_grafico_comparativo(df_real)
                if fig:
                    self._mostrar_grafico(fig)
                    
                    # Mostrar métricas
                    metricas = self.optimizador.comparar_real_estimado(df_real)
                    messagebox.showinfo("Métricas de Comparación",
                        f"Error medio: {metricas['error_medio']:.2f}\n"
                        f"Error porcentual: {metricas['error_porcentual']:.1f}%\n"
                        f"Productos afectados: {metricas['productos_afectados']}"
                    )
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
            
    def _mostrar_grafico(self, fig):
        """Muestra un gráfico de plotly en el canvas"""
        # Convertir gráfico a imagen
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

def main():
    root = tk.Tk()
    app = SistemaStock(root)
    root.mainloop()

if __name__ == "__main__":
    main()