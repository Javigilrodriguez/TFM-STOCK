import pandas as pd
import numpy as np

class Utilidades:
    def __init__(self):
        pass

    def leer_excel(self, ruta_archivo):
        """Lee archivo Excel con manejo específico para cada tipo de archivo"""
        try:
            # Determinar si es un archivo de planning o dataset
            if 'planning' in ruta_archivo.lower():
                return self._leer_planning(ruta_archivo)
            else:
                return self._leer_dataset(ruta_archivo)
        except Exception as e:
            print(f"Error leyendo Excel {ruta_archivo}: {e}")
            raise

    def _leer_dataset(self, ruta_archivo):
        """Lee archivos de dataset con estructura específica"""
        try:
            # Leer primeras filas para analizar estructura
            df = pd.read_excel(ruta_archivo, nrows=5)
            
            # Verificar si tenemos las filas correctas
            if 'Fecha' in df.columns:
                # Leer el archivo completo saltando las filas de encabezado
                df = pd.read_excel(ruta_archivo, skiprows=5)
                
                # Asignar nombres de columnas correctos
                nombres_columnas = [
                    'COD_ART',        # Primera columna
                    'NOMBRE',         # Segunda columna
                    'Cj/H',          # Cajas por hora
                    'Disponible',     # Stock disponible
                    'Calidad',        # Stock en calidad
                    'Stock Externo',  # Stock externo
                    'M_Vta -15',      # Ventas últimos 15 días
                    'M_Vta -15 AA'    # Ventas año anterior
                ]
                
                # Asignar nombres solo a las columnas que necesitamos
                if len(df.columns) >= len(nombres_columnas):
                    df = df.iloc[:, :len(nombres_columnas)]
                    df.columns = nombres_columnas
                else:
                    df.columns = nombres_columnas[:len(df.columns)]
                
                # Eliminar filas que no tienen código de artículo
                df = df.dropna(subset=[df.columns[0]])
                
                return df
            else:
                raise ValueError("Estructura de archivo no reconocida")
                
        except Exception as e:
            print(f"Error en lectura de dataset: {e}")
            raise

    def _leer_planning(self, ruta_archivo):
        """Lee archivo de planning con estructura específica"""
        try:
            df = pd.read_excel(ruta_archivo)
            
            # Restructurar el planning para formato estándar
            columnas_por_dia = ['H', 'CJ', 'Cob', 'Línea']
            dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sábado']
            
            planning_procesado = pd.DataFrame()
            
            # Procesar cada día
            for dia in dias:
                dia_data = df[df.iloc[:, 1] == dia].copy()
                if not dia_data.empty:
                    dia_data = dia_data.iloc[:, :len(columnas_por_dia)]
                    dia_data.columns = columnas_por_dia
                    dia_data['DIA'] = dia
                    planning_procesado = pd.concat([planning_procesado, dia_data])
            
            return planning_procesado
            
        except Exception as e:
            print(f"Error en lectura de planning: {e}")
            raise

    def procesar_datos(self, df):
        """
        Limpia y prepara los datos para el análisis.
        Redondea hacia arriba los valores numéricos.
        """
        try:
            # Eliminar filas completamente vacías
            df = df.dropna(how='all')
            
            # Convertir columnas numéricas y redondear hacia arriba
            columnas_numericas = [
                'Cj/H', 'Disponible', 'Calidad', 
                'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA'
            ]
            
            for col in columnas_numericas:
                if col in df.columns:
                    # Convertir a numérico, manteniendo NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Redondear hacia arriba y convertir a entero
                    df[col] = df[col].apply(lambda x: int(np.ceil(x)) if pd.notnull(x) else x)
            
            # Valores por defecto
            valores_default = {
                'Cj/H': 1,
                'Disponible': 0,
                'Calidad': 0,
                'Stock Externo': 0,
                'M_Vta -15': 0,
                'M_Vta -15 AA': 1
            }
            df = df.fillna(valores_default)
            
            # Stock total como suma de los stocks (ya redondeados)
            df['STOCK_TOTAL'] = (df['Disponible'] + 
                               df['Calidad'] + 
                               df['Stock Externo']).astype(int)
            
            # Días de cobertura redondeados hacia arriba
            venta_diaria = np.where(df['M_Vta -15'] > 0, df['M_Vta -15'] / 15, 1)
            df['DIAS_COBERTURA'] = (df['STOCK_TOTAL'] / venta_diaria).apply(
                lambda x: int(np.ceil(x)) if x != np.inf else 30
            )
            
            # Asegurar que todos los valores numéricos sean enteros
            columnas_a_entero = columnas_numericas + ['STOCK_TOTAL', 'DIAS_COBERTURA']
            for col in columnas_a_entero:
                if col in df.columns:
                    df[col] = df[col].astype(int)
            
            print("\nEjemplo de datos procesados:")
            print(df[columnas_a_entero].head())
            
            return df
            
        except Exception as e:
            print(f"Error procesando datos: {e}")
            print(f"Columnas disponibles: {df.columns.tolist()}")
            raise

    def validar_datos(self, df):
        """Verifica que los datos cumplan con los requisitos mínimos"""
        try:
            # Verificar columnas requeridas
            columnas_requeridas = [
                'COD_ART', 'Cj/H', 'Disponible', 'Calidad', 
                'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA'
            ]
            
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            if faltantes:
                print(f"Faltan columnas requeridas: {faltantes}")
                print("Columnas disponibles:", df.columns.tolist())
                return False
            
            # Verificar que hay datos válidos
            if len(df) == 0:
                print("El dataset está vacío")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación: {e}")
            return False