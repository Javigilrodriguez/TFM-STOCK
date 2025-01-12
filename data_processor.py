# data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """Procesador de datos para el sistema de gestión de stock"""
    
    @staticmethod
    def clean_data(df):
        """Limpia y prepara los datos"""
        try:
            # Eliminar filas vacías
            df.dropna(how='all', inplace=True)
            
            # Convertir columnas numéricas
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
                         'M_Vta -15', 'M_Vta -15 AA']
            
            for col in num_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].replace({',': '.'}, regex=True), 
                                          errors='coerce')
                    print(f"Columna {col} convertida a numérica")
            
            # Rellenar valores nulos
            df.fillna({
                'Cj/H': 1,
                'Disponible': 0,
                'Calidad': 0,
                'Stock Externo': 0,
                'M_Vta -15': 0,
                'M_Vta -15 AA': 1
            }, inplace=True)
            
            # Calcular stock total
            df['STOCK_TOTAL'] = (df['Disponible'] + 
                                df['Calidad'] + 
                                df['Stock Externo'])
            
            # Calcular métricas adicionales
            if 'M_Vta -15' in df.columns and 'M_Vta -15 AA' in df.columns:
                df['RATIO_CAMBIO'] = df['M_Vta -15'] / df['M_Vta -15 AA']
                df['DIAS_COBERTURA'] = df['STOCK_TOTAL'] / (df['M_Vta -15'] / 15)
                
                # Limpiar ratios infinitos
                df['RATIO_CAMBIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
                df['DIAS_COBERTURA'].replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Rellenar con medias
                df['RATIO_CAMBIO'].fillna(df['RATIO_CAMBIO'].mean(), inplace=True)
                df['DIAS_COBERTURA'].fillna(df['DIAS_COBERTURA'].mean(), inplace=True)
            
            print(f"Datos limpios: {len(df)} productos activos")
            return df

        except Exception as e:
            print(f"Error en limpieza de datos: {e}")
            raise

    @staticmethod
    def validate_data(df):
        """Valida la integridad y coherencia de los datos"""
        try:
            required_columns = [
                'COD_ART', 'Cj/H', 'Disponible', 'Calidad', 
                'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA'
            ]
            
            # Verificar columnas requeridas
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
            
            # Verificar valores negativos
            negative_stocks = df[df[['Disponible', 'Calidad', 'Stock Externo']] < 0].any(axis=1)
            if negative_stocks.any():
                print("ADVERTENCIA: Existen valores negativos en stocks")
                
            # Verificar producción por hora
            invalid_prod = df[df['Cj/H'] <= 0]
            if not invalid_prod.empty:
                print("ADVERTENCIA: Existen tasas de producción inválidas")
            
            return True

        except Exception as e:
            print(f"Error en validación de datos: {e}")
            return False
            
    @staticmethod
    def calculate_tendencies(df):
        """Calcula tendencias y variaciones estacionales"""
        try:
            if 'M_Vta -15' in df.columns and 'M_Vta -15 AA' in df.columns:
                # Calcular variación respecto al año anterior
                df['VAR_ANUAL'] = ((df['M_Vta -15'] - df['M_Vta -15 AA']) / 
                                df['M_Vta -15 AA'] * 100)
                
                # Identificar productos con variación significativa (>20%)
                significant_var = df[abs(df['VAR_ANUAL']) > 20]
                
                print("\nAnálisis de tendencias:")
                print(f"Productos con variación significativa: {len(significant_var)}")
                print(f"Variación media: {df['VAR_ANUAL'].mean():.2f}%")
                
            return df

        except Exception as e:
            print(f"Error en cálculo de tendencias: {e}")
            raise