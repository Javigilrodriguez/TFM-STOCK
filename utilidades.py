import pandas as pd
import numpy as np
import os
import re
from typing import Optional

class Utilidades:
    def __init__(self):
        self.debug = True

    def cargar_datasets(self, carpeta: str) -> pd.DataFrame:
        """
        Carga todos los datasets CSV de la carpeta seleccionada
        
        Args:
            carpeta: Ruta a la carpeta con los archivos CSV
            
        Returns:
            DataFrame combinado con todos los datos
        """
        try:
            if not os.path.exists(carpeta):
                raise ValueError(f"La carpeta {carpeta} no existe")
            
            # Buscar todos los archivos CSV en la carpeta
            archivos_csv = [f for f in os.listdir(carpeta) if f.lower().endswith('.csv')]
            
            if not archivos_csv:
                raise ValueError("No se encontraron archivos CSV en la carpeta")
            
            datasets = []
            
            for archivo in archivos_csv:
                ruta_completa = os.path.join(carpeta, archivo)
                try:
                    # Intentar leer el CSV con diferentes encodings
                    df = None
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            df = pd.read_csv(ruta_completa, sep=';', encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        print(f"No se pudo leer el archivo {archivo} con ningún encoding")
                        continue
                    
                    # Cargar el archivo si no está vacío
                    if not df.empty:
                        datasets.append(df)
                        print(f"Archivo cargado: {archivo} ({len(df)} registros)")
                    
                except Exception as e:
                    print(f"Error procesando {archivo}: {str(e)}")
            
            if not datasets:
                raise ValueError("No se pudieron cargar datos válidos de ningún archivo")
            
            # Combinar datasets
            df_combinado = pd.concat(datasets, ignore_index=True)
            
            return df_combinado
            
        except Exception as e:
            raise Exception(f"Error cargando datasets: {str(e)}")

    def procesar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa y limpia los datos del DataFrame
        
        Args:
            df: DataFrame a procesar
            
        Returns:
            DataFrame procesado
        """
        try:
            df_proc = df.copy()
            
            def limpiar_valor_numerico(valor):
                """Limpia un valor numérico con posible formato español"""
                if pd.isna(valor) or valor == '' or valor == '(en blanco)':
                    return 0
                if isinstance(valor, (int, float)):
                    return valor
                try:
                    # Limpiar string
                    valor_str = str(valor).strip()
                    valor_str = valor_str.replace(',', '.').replace(' ', '')
                    return float(valor_str)
                except:
                    return 0
            
            # Columnas numéricas puras (solo números)
            columnas_numericas = [
                'Disponible', 'Calidad', 'Pedido', 
                'Vta -15', 'Vta -60', 'Vta -15 AA', 'Vta +15 AA'
            ]
            
            # Columnas con posible formato especial
            columnas_especiales = [
                'Cj/H', 'Stock Externo', 'M_Vta -15', 'M_Vta-60',
                'M_Vta -15 AA', 'M_Vta +15 AA', '% Vta-15/ AA', '% Vta-15/ +15 AA'
            ]
            
            # Procesar columnas numéricas
            for col in columnas_numericas:
                if col in df_proc.columns:
                    df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
            
            # Procesar columnas especiales
            for col in columnas_especiales:
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].apply(limpiar_valor_numerico)
            
            # Mantener columnas de texto como están
            columnas_texto = ['COD_ART', 'NOM_ART', 'COD_GRU', '1ª OF', 'OF']
            for col in columnas_texto:
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].fillna('')
            
            # Calcular columnas adicionales
            df_proc['STOCK_TOTAL'] = (
                df_proc['Disponible'] + 
                df_proc['Calidad'] + 
                df_proc['Stock Externo']
            )
            
            if self.debug:
                print("\nTipos de datos después de procesar:")
                print(df_proc.dtypes)
                print("\nMuestra de valores procesados:")
                print(df_proc[['Cj/H', 'M_Vta -15', 'STOCK_TOTAL']].head())
            
            return df_proc
            
        except Exception as e:
            print(f"Error procesando datos: {str(e)}")
            raise

    def validar_datos(self, df: pd.DataFrame) -> bool:
        """
        Valida que el DataFrame tenga la estructura requerida
        
        Args:
            df: DataFrame a validar
            
        Returns:
            bool: True si los datos son válidos
        """
        try:
            # Verificar columnas requeridas
            columnas_req = ['COD_ART', 'NOM_ART', 'Cj/H', 'Disponible', 
                          'Calidad', 'Stock Externo']
            
            faltantes = [col for col in columnas_req if col not in df.columns]
            if faltantes:
                print(f"Faltan columnas requeridas: {faltantes}")
                return False
            
            # Verificar datos no vacíos
            if df.empty:
                print("El DataFrame está vacío")
                return False
            
            # Verificar tipos de datos
            if not pd.api.types.is_numeric_dtype(df['Cj/H']):
                print("La columna Cj/H debe ser numérica")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación: {str(e)}")
            return False

    def limpiar_nombre_columna(self, nombre: str) -> str:
        """
        Limpia y estandariza el nombre de una columna
        
        Args:
            nombre: Nombre original de la columna
            
        Returns:
            str: Nombre limpio
        """
        try:
            # Quitar caracteres especiales y espacios
            limpio = re.sub(r'[^\w\s-]', '', nombre)
            limpio = limpio.strip().replace(' ', '_')
            return limpio
            
        except Exception as e:
            print(f"Error limpiando nombre de columna: {str(e)}")
            return nombre