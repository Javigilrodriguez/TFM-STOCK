import pandas as pd
import numpy as np
import os
import re

class Utilidades:
    def __init__(self):
        pass

    def limpiar_nombre_columna(self, nombre):
        """
        Limpia y normaliza el nombre de una columna
        - Elimina caracteres especiales
        - Convierte a mayúsculas
        - Quita espacios
        """
        limpio = re.sub(r'[/\-_]', '', nombre.upper().strip())
        return limpio

    def leer_csv(self, ruta_archivo):
        """Lee y procesa un archivo CSV con estructura compleja."""
        try:
            # Leer todas las líneas del archivo
            with open(ruta_archivo, 'r', encoding='cp1252') as f:
                lineas = f.readlines()
            
            # Encontrar el índice de la línea con COD_ART
            indice_encabezados = None
            for i, linea in enumerate(lineas):
                if 'COD_ART' in linea:
                    indice_encabezados = i
                    break
            
            if indice_encabezados is None:
                raise ValueError("No se encontró la línea de encabezados con COD_ART")
            
            # Leer el DataFrame desde la línea de encabezados
            df = pd.read_csv(
                ruta_archivo, 
                sep=';', 
                encoding='cp1252', 
                header=indice_encabezados,  # Use the line with headers
                skip_blank_lines=True,
                dtype=str  # Read everything as string initially
            )
            
            # Columnas requeridas con sus posibles variantes
            df.columns = [col.strip() for col in df.columns]  # Remove any whitespace
            
            # Convertir columnas numéricas con manejo de valores especiales
            columnas_numericas = [
                'Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
                'VTA -15', 'M_Vta -15'
            ]
            
            for col in columnas_numericas:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .replace('(en blanco)', '0')
                        .str.replace(',', '.')  # Replace comma with dot for decimal
                        .str.replace(' ', '')   # Remove spaces
                    )
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
        
        except Exception as e:
            print(f"Error procesando archivo {ruta_archivo}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def procesar_datos(self, datos):
        """
        Procesa datos de un DataFrame o de un archivo dado.
        """
        try:
            # Si la entrada es una ruta de archivo
            if isinstance(datos, str):
                if datos.endswith('.csv'):
                    df = self.leer_csv(datos)
                else:
                    raise ValueError("Formato de archivo no soportado. Use CSV.")
            # Si la entrada ya es un DataFrame
            elif isinstance(datos, pd.DataFrame):
                df = datos
            else:
                raise ValueError("El argumento debe ser una ruta de archivo o un DataFrame.")

            # Renombrar columnas para cumplir con los requisitos del sistema
            columnas_mapeo = {
                'Cj/H': 'Cj/H',
                'Disponible': 'DISPONIBLE',
                'Calidad': 'CALIDAD',
                'Stock Externo': 'STOCK EXTERNO',
                'VTA -15': 'VTA_-15',
                'M_Vta -15': 'M_Vta -15'
            }
            
            # Renombrar columnas
            df.rename(columns=columnas_mapeo, inplace=True)
            
            # Asegurar que se mantengan las columnas necesarias
            columnas_necesarias = [
                'COD_ART', 'NOM_ART', 'COD_GRU', 
                'Cj/H', 'DISPONIBLE', 'CALIDAD', 'STOCK EXTERNO'
            ]
            
            # Calcular métricas adicionales
            df['STOCK_TOTAL'] = df['DISPONIBLE'] + df['CALIDAD'] + df['STOCK EXTERNO']
            
            # Calcular días de cobertura
            df['DIAS_COBERTURA'] = np.where(
                df['M_Vta -15'] > 0,
                df['STOCK_TOTAL'] / df['M_Vta -15'],
                float('inf')
            )

            return df

        except Exception as e:
            print(f"Error procesando datos: {e}")
            raise

    def validar_datos(self, df):
        """Valida que el DataFrame cumpla con los requisitos mínimos."""
        try:
            columnas_requeridas = [
                'COD_ART', 'NOM_ART', 'COD_GRU', 'Cj/H', 
                'DISPONIBLE', 'CALIDAD', 'STOCK EXTERNO'
            ]

            # Verificar columnas faltantes
            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            if faltantes:
                print(f"Faltan columnas requeridas: {faltantes}")
                return False

            # Verificar si el DataFrame está vacío
            if df.empty:
                print("El DataFrame está vacío.")
                return False

            return True
        except Exception as e:
            print(f"Error validando datos: {e}")
            return False