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
        """
        try:
            if not os.path.exists(carpeta):
                raise ValueError(f"La carpeta {carpeta} no existe")
                
            # Buscar archivos CSV
            archivos_csv = [f for f in os.listdir(carpeta) if f.lower().endswith('.csv')]
            
            if not archivos_csv:
                raise ValueError("No se encontraron archivos CSV en la carpeta")
                
            datasets = []
            
            for archivo in archivos_csv:
                ruta_completa = os.path.join(carpeta, archivo)
                try:
                    # Intentar diferentes encodings
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            # Leer primero unas pocas líneas para detectar el separador
                            with open(ruta_completa, 'r', encoding=encoding) as f:
                                primeras_lineas = [next(f) for _ in range(5)]
                                
                            # Detectar separador
                            separador = ';' if ';' in primeras_lineas[0] else ','
                            
                            # Leer el archivo completo
                            df = pd.read_csv(ruta_completa, sep=separador, encoding=encoding)
                            
                            # Verificar y renombrar columnas si es necesario
                            df.columns = [col.strip() for col in df.columns]
                            
                            # Mapeo de nombres de columnas alternativos
                            mapeo_columnas = {
                                'CODIGO': 'COD_ART',
                                'NOMBRE': 'NOM_ART',
                                'STOCK_DISPONIBLE': 'Disponible',
                                'CAJAS_HORA': 'Cj/H'
                            }
                            
                            # Renombrar columnas si es necesario
                            df = df.rename(columns=mapeo_columnas)
                            
                            if not df.empty:
                                datasets.append(df)
                                print(f"Archivo cargado: {archivo} ({len(df)} registros)")
                            break
                            
                        except UnicodeDecodeError:
                            continue
                            
                except Exception as e:
                    print(f"Error procesando {archivo}: {str(e)}")
                    continue
                    
            if not datasets:
                raise ValueError("No se pudieron cargar datos válidos de ningún archivo")
                
            # Combinar datasets
            df_combinado = pd.concat(datasets, ignore_index=True)
            
            # Verificar y mostrar información
            print("\nInformación del dataset combinado:")
            print(f"Registros totales: {len(df_combinado)}")
            print("Columnas encontradas:", df_combinado.columns.tolist())
            print("\nPrimeras filas:")
            print(df_combinado.head())
            
            return df_combinado
            
        except Exception as e:
            raise Exception(f"Error cargando datasets: {str(e)}")

    def procesar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_proc = df.copy()
            
            def limpiar_valor_numerico(valor):
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
            
            # Procesar columnas numéricas
            columnas_numericas = [
                'Disponible', 'Calidad', 'Stock Externo', 
                'M_Vta -15', 'M_Vta -15 AA'
            ]
            
            for col in columnas_numericas:
                if col in df_proc.columns:
                    df_proc[col] = df_proc[col].apply(limpiar_valor_numerico)
            
            # Calcular columnas adicionales de forma robusta
            df_proc['STOCK_TOTAL'] = (
                df_proc.get('Disponible', 0) + 
                df_proc.get('Calidad', 0) + 
                df_proc.get('Stock Externo', 0)
            )
            
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