# Wine Classification with PySpark and MLflow

## Descripción

Este proyecto se enfoca en la clasificación de vinos utilizando PySpark para el procesamiento de datos y modelos de aprendizaje automático. El seguimiento y la gestión de experimentos se realiza con MLflow, lo que permite un registro detallado de los resultados obtenidos durante el proceso de entrenamiento.

## Requisitos

- Python 3.8 o superior
- PySpark
- MLflow
- Pandas
- Numpy

## Instalación

Para instalar las dependencias necesarias, puedes utilizar pip:

```bash
pip install pyspark mlflow pandas numpy
```
## Crear experientos en mlflow
```bash
import mlflow
from mlflow import spark

# Configurar el experimento en MLflow
mlflow.set_experiment('/Users/andrescandelo17@gmail.com/wine-classification-experiment')
```

## Carga de Datos y transformar los datos
Primero, los datos se cargan y preparan para el modelado. Se agrega nombres a las columnas con base en el diccionario de datos enviado
```bash
from pyspark.sql import SparkSession

# Inicializar sesión de Spark
spark = SparkSession.builder.appName("WineClassification").getOrCreate()

# Cargar datos
data = spark.read.csv("winedata.csv", header=True, inferSchema=True)


# los nombres de las columnas son
column_names = [
    "Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
    "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
    "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline"
]

```


