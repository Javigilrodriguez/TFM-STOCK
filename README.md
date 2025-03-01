# Sistema de Planificación de Producción

## Descripción
Sistema de optimización para planificación de producción que integra predicción de demanda, gestión de calendario de producción y método Simplex para optimización de recursos.

## Características
- Predicción de demanda usando regresión lineal
- Calendario interactivo para gestión de días hábiles
- Optimización de producción mediante método Simplex
- Interfaz gráfica intuitiva
- Persistencia de configuraciones y modelos
- Generación de reportes detallados

## Requisitos del Sistema
- Python 3.8+
- 2GB RAM mínimo
- 500MB espacio en disco

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/planificacion-produccion.git
cd planificacion-produccion
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso del Sistema

### Inicio del Sistema
```bash
python main.py
```

### Configuración Inicial
1. Seleccionar carpeta de datos
2. Configurar calendario de producción
3. Ajustar parámetros de producción:
   - Días de stock seguridad
   - Horas mínimas producción
   - Horas disponibles
   - Horas mantenimiento

### Gestión del Calendario
- Seleccionar días en el calendario
- Marcar/desmarcar días hábiles
- La configuración se guarda automáticamente

### Entrenamiento del Modelo
1. Seleccionar datos de entrenamiento
2. Ejecutar entrenamiento
3. El modelo se guarda automáticamente

### Generación de Plan
1. Seleccionar rango de fechas
2. Ajustar parámetros si es necesario
3. Ejecutar optimización
4. Revisar y guardar plan generado

## Estructura del Proyecto
```
proyecto/
├── main.py              # Interfaz principal
├── modelo.py            # Implementación del modelo
├── optimizador.py       # Sistema de optimización
├── utilidades.py        # Funciones auxiliares
├── calendario.py        # Gestión de calendario
├── requirements.txt     # Dependencias
└── modelos/            # Directorio de modelos
```

## Archivos de Entrada

### Formato de Datos
El sistema acepta archivos CSV y Excel con la siguiente estructura:
```
COD_ART     : Código de artículo (str)
Cj/H        : Cajas por hora (float)
Disponible  : Stock disponible (int)
Calidad     : Stock en calidad (int)
Stock Externo: Stock en almacenes externos (int)
M_Vta -15   : Ventas últimos 15 días (int)
M_Vta -15 AA: Ventas año anterior (int)
```

## Salidas del Sistema

### Plan de Producción
Archivo CSV con:
- Datos originales
- Predicciones de demanda
- Plan de producción
- Métricas y KPIs

## Métricas de Rendimiento
- MAE típico: 21.29
- RMSE: 24.45
- R²: 0.99
- Tiempo procesamiento: <1s para 50 productos

## Limitaciones
- Máximo 200 productos simultáneos
- Predicciones hasta 15 días
- Requiere datos históricos
- Mínimo 2 horas por lote
