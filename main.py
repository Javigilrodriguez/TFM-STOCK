import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import os

from modelo import ModeloPrediccion
from optimizador import Optimizador
from utilidades import Utilidades

class SistemaStock:
    def __init__(self, root):
        """Inicializa la aplicación"""
        self.root = root
        self.root.title("Sistema de Planificación de Producción")
        self.root.geometry("800x600")
        
        # Variables de control
        self.ruta_carpeta = tk.StringVar()
        self.modelo_cargado = tk.StringVar(value="Estado: No hay modelo cargado")
        self.modelo_entrenado = False
        
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
            modelo_actual = modelos[0]  # El más reciente
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
    
    def seleccionar_carpeta(self):
        """Permite al usuario seleccionar una carpeta con datasets"""
        carpeta = filedialog.askdirectory(title="Seleccionar Carpeta con Datasets")
        if carpeta:
            self.ruta_carpeta.set(carpeta)
            self.status_label.config(text="Carpeta seleccionada. Puede proceder con el entrenamiento.")
    
    def cargar_datasets(self):
        """Carga todos los datasets de la carpeta seleccionada"""
        carpeta = self.ruta_carpeta.get()
        if not carpeta:
            raise ValueError("Debe seleccionar una carpeta primero")
        
        datasets = []
        archivos_validos = [f for f in os.listdir(carpeta) 
                           if f.endswith(('.xlsx', '.xls', '.csv'))]
        
        for archivo in archivos_validos:
            ruta_completa = os.path.join(carpeta, archivo)
            try:
                if archivo.endswith('.csv'):
                    df = pd.read_csv(ruta_completa, sep=';', encoding='latin1')
                else:
                    df = self.utilidades.leer_excel(ruta_completa)
                
                if df is not None and not df.empty:
                    datasets.append(df)
                    print(f"Archivo cargado: {archivo}")
            
            except Exception as e:
                print(f"Error cargando {archivo}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No se encontraron datos válidos en los archivos")
        
        # Combinar todos los datasets
        df_combinado = pd.concat(datasets, ignore_index=True)
        
        print(f"\nColumnas encontradas:")
        print(df_combinado.columns.tolist())
        print(f"Total de registros: {len(df_combinado)}")
        
        return df_combinado
    
    def entrenar_modelo(self):
        """Entrena el modelo con los datos seleccionados"""
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
            y = df_procesado['M_Vta -15']
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
            # Cargar datos actuales
            df_actual = self.cargar_datasets()
            df_procesado = self.utilidades.procesar_datos(df_actual)
            
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
            
            # Guardar resultados
            ruta_salida = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="plan_produccion.csv"
            )
            
            if ruta_salida:
                df_procesado.to_csv(ruta_salida, index=False, sep=';')
                
                # Mostrar resumen
                total_cajas = df_procesado['CAJAS_PRODUCIR'].sum()
                total_horas = df_procesado['HORAS_PRODUCCION'].sum()
                productos_programados = (df_procesado['CAJAS_PRODUCIR'] > 0).sum()
                
                mensaje = (f"Plan de producción generado exitosamente\n\n"
                          f"Productos programados: {productos_programados}\n"
                          f"Total cajas: {total_cajas:.0f}\n"
                          f"Total horas: {total_horas:.1f}")
                
                messagebox.showinfo("Éxito", mensaje)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en optimización: {str(e)}")

def main():
    try:
        root = tk.Tk()
        app = SistemaStock(root)
        root.mainloop()
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")

if __name__ == "__main__":
    main()