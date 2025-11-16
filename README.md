# PROYECTO-PYTHON
"""
CAMPO                | TIPO DE DATO | DESCRIPCIÓN
---------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------
ANNO                 | Entero       | Corresponde al año para el cual se reporta la información.                                                                 |
TRIMESTRE            | Entero       | Corresponde al trimestre del año para el cual se reporta la información.                                                   |
ID_DEPARTAMENTO      | Entero       | Codificación de la DIVIPOLA (DANE) para los departamentos.                                                                 |        
DEPARTAMENTO         | Texto        | Nombre del departamento.                                                                                                   |
ID_MUNICIPIO         | Entero       | Codificación de la DIVIPOLA (DANE) para los municipios.                                                                    |
MUNICIPIO            | Texto        | Nombre del municipio.                                                                                                      |
ID_CPOB              | Entero       | Codificación de la DIVIPOLA (DANE) para cabeceras municipales y centros poblados.                                          |
CPOB                 | Texto        | Nombre de la cabecera municipal o centro poblado.                                                                          |
AREA_CPOB            | Flotante     | Área en km² de la cabecera municipal o centro poblado según el Mapa Geoestadístico Nacional 2023 del DANE.                 |
ID_TECNOLOGIA        | Entero       | Código de tecnología: 2=2G, 3=3G, 4=4G, 5=5G. En zonas sin cobertura el valor es 0.                                        |
TECNOLOGIA           | Texto        | Nombre de la tecnología asociada (2G, 3G, 4G, 5G). En zonas sin cobertura el valor es “Ninguna”.                           |       
NIVEL_SENAL          | Entero       | Nivel de potencia de recepción según la tabla del PRSTM (Circular CRC 156 de 2024). En zonas sin cobertura el valor es 0.  |
AREA_COB_CLARO       | Flotante     | Área en km² cubierta por CLARO.                                                                                            |
AREA_COB_MOVISTAR    | Flotante     | Área en km² cubierta por MOVISTAR.                                                                                         |
AREA_COB_TIGO        | Flotante     | Área en km² cubierta por TIGO.                                                                                             |
AREA_COB_WOM         | Flotante     | Área en km² cubierta por WOM.                                                                                              |
---------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd

# leer base de datos
df = pd.read_csv('./data/Datos_Cobertura Movil_1T_2023 a 4T_2024.csv', sep=';')
df.info()

# Dimensiones del df
print(df.shape)

# Revisar si hay valores nulos
df.isnull().values.any()
print(df.shape)

# Revisar los tipos de datos
print(df.dtypes)

# Convertir el tipo de dato de cada variable
cols = ['AREA_COB_CLARO', 'AREA_COB_MOVISTAR', 'AREA_COB_TIGO', 'AREA_COB_WOM','AREA_CPOB']

df[cols] = (
    df[cols]
    .apply(lambda x: x.str.replace(',', '.', regex=False))  # Reemplaza coma por punto
    .astype(float)  # Convertir a número decimal
)

# Convertir el tipo de dato de cada variable
df['ANNO'] = df['ANNO'].astype(str)
df['TRIMESTRE'] = df['TRIMESTRE'].astype(str)
df['ID_DEPARTAMENTO'] = df['ID_DEPARTAMENTO'].astype(str)
df['DEPARTAMENTO'] = df['DEPARTAMENTO'].astype(str)
df['ID_MUNICIPIO'] = df['ID_MUNICIPIO'].astype(str)
df['MUNICIPIO'] = df['MUNICIPIO'].astype(str)
df['CPOB'] = df['CPOB'].astype(str)
df['ID_TECNOLOGIA'] = df['ID_TECNOLOGIA'].astype(str)

print(df.dtypes)

# Resumen estadístico
df.describe()

# Contar el número de departamentos únicos
num_departamentos = df['DEPARTAMENTO'].nunique()
print(f"Departamentos: {num_departamentos}")

# Contar el número de municipios únicos
num_municipios = df['MUNICIPIO'].nunique()
print(f"Municipios: {num_municipios}")

# Contar el número de CPOB únicos
num_cpob = df['CPOB'].nunique()
print(f"CPOB: {num_cpob}")

cols = ['AREA_COB_CLARO', 'AREA_COB_MOVISTAR', 'AREA_COB_TIGO', 'AREA_COB_WOM']

#---------------------------------------------

# Filtrar datos para el año 2024 y trimestre 4

df_actual = df[(df['ANNO'] == '2024') & (df['TRIMESTRE'] == '4')]


# Agrupar por DEPARTAMENTO, MUNICIPIO, CPOB y TECNOLOGIA y sumar las áreas de cobertura
df_actual = (
    df_actual.groupby(['ANNO','TRIMESTRE','DEPARTAMENTO','MUNICIPIO','CPOB','TECNOLOGIA'], as_index=False)
    .agg({
        'AREA_CPOB': 'first',  # el área total urbana es la misma
        'AREA_COB_CLARO': 'sum',
        'AREA_COB_MOVISTAR': 'sum',
        'AREA_COB_TIGO': 'sum',
        'AREA_COB_WOM': 'sum'
        })
)


# Calcular el área de cobertura máxima entre los operadores para cada fila
df_actual["AREA_COB_MAX"] = df_actual[["AREA_COB_CLARO", "AREA_COB_MOVISTAR", "AREA_COB_TIGO", "AREA_COB_WOM"]].max(axis=1)

# Identificar de qué operador es ese máximo
df_actual["OPERADOR_MAX"] = (
    df_actual[["AREA_COB_CLARO", "AREA_COB_MOVISTAR", "AREA_COB_TIGO", "AREA_COB_WOM"]]
    .idxmax(axis=1)                       # obtiene el nombre de la columna con el valor máximo
    .str.replace("AREA_COB_", "")          # quita el prefijo
    .str.upper()                           # pone todo en mayúsculas (por estética)
)

# Calcular el máximo y la tecnología correspondiente por cada CPOB
df_max_tecnologia = (
    df_actual.loc[df_actual.groupby(['ANNO','TRIMESTRE','DEPARTAMENTO','MUNICIPIO','CPOB'])['AREA_COB_MAX'].idxmax()]
    .copy()
)

# Renombrar la columna para mayor claridad
df_max_tecnologia.rename(columns={'AREA_COB_MAX': 'AREA_COB_MAX_TECNOLOGIAS'}, inplace=True)

# Crear una columna que identifique la tecnología del máximo
df_max_tecnologia['TECNOLOGIA_MAX'] = df_max_tecnologia['TECNOLOGIA']

# Calcular el porcentaje de cobertura del operador con mayor área de cobertura en comparación con el área total del CPOB
df_max_tecnologia['PORCENTAJE_COBERTURA'] = (df_max_tecnologia['AREA_COB_MAX_TECNOLOGIAS'] / df_max_tecnologia['AREA_CPOB']) * 100
df_max_tecnologia = df_max_tecnologia.sort_values(by='PORCENTAJE_COBERTURA',ascending=True)


# Agrupar por departamento
df_departamento = df_max_tecnologia.groupby('DEPARTAMENTO', as_index=False).agg({
    'PORCENTAJE_COBERTURA': 'mean',
    'OPERADOR_MAX': lambda x: x.value_counts().idxmax()
})

# Top 10 con menor y mayor cobertura
top10_menor = df_departamento.sort_values(by='PORCENTAJE_COBERTURA', ascending=True).head(6)
top10_mayor = df_departamento.sort_values(by='PORCENTAJE_COBERTURA', ascending=False).head(6)

# Añadir signo negativo a los de menor cobertura (para el espejo)
top10_menor['PORCENTAJE_COBERTURA'] = -top10_menor['PORCENTAJE_COBERTURA']

# Unir ambos en un solo DataFrame
df_comparativo = pd.concat([top10_menor, top10_mayor])

# Contar número de CPOB por operador predominante
conteo_operador = df_max_tecnologia['OPERADOR_MAX'].value_counts()

# Calcular porcentaje
porcentaje_operador = (conteo_operador / conteo_operador.sum()) * 100


# Sumar área ganadora por operador en cada municipio
df_municipio = (
    df_max_tecnologia
    .groupby(['DEPARTAMENTO', 'MUNICIPIO', 'OPERADOR_MAX'], as_index=False)
    .agg({'AREA_COB_MAX_TECNOLOGIAS': 'sum'})
)

# Para cada municipio seleccionar el operador que mayor área suma
df_municipio_predominante = (
    df_municipio.loc[
        df_municipio.groupby(['DEPARTAMENTO', 'MUNICIPIO'])['AREA_COB_MAX_TECNOLOGIAS'].idxmax()
    ][['DEPARTAMENTO', 'MUNICIPIO', 'OPERADOR_MAX']]
)

# Filtrar por año 2024 y trimestre 4
df_filtrado = df[(df['ANNO'] == '2024') & (df['TRIMESTRE'] == '4')]

# Seleccionar los top departamentos por cantidad de registros
top_deptos = df_filtrado['DEPARTAMENTO'].value_counts().head(30).index
df_top = df_filtrado[df_filtrado['DEPARTAMENTO'].isi

# Gráficos de Análisis de Cobertura Móvil

# Importar librerías
import matplotlib.pyplot as plt, \
       matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from .code import (
    df, df_max_tecnologia, df_comparativo, 
    porcentaje_operador, conteo_operador, cols, 
    df_top
)

# Colores por operador (definición centralizada)
color_dict = {
    'CLARO': '#ED1B24',      # rojo
    'MOVISTAR': '#66CD00',   # azul
    'TIGO': '#001EB4',       # verde
    'WOM': '#6F1A7F',        # morado
    'OTRO': "#949494"        # gris
}

# Definir colores personalizados
colores = {
    '2G': 'red',
    '3G': 'orange',
    '4G': 'green',
    '5G': 'blue',
    'Ninguna': 'gray'  # Color para los CPOB sin tecnología
}

#------- GRAFICO 1 -------#

plt.figure(figsize=(14, 6))
sns.countplot(
    data=df_top,
    x='DEPARTAMENTO',
    hue='TECNOLOGIA',
    palette=colores
)
plt.title('Distribución de Tecnologías en Top Departamentos (2024 - Trimestre 4)', fontsize=14, fontweight='bold')
plt.xlabel('Departamento', fontsize=12)
plt.ylabel('Cantidad de Registros', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Tecnología', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#------- GRAFICO 2 -------#

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[cols])

plt.title('Distribución del Área de Cobertura por Operador')
plt.xlabel('Operador')
plt.ylabel('Área de Cobertura')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#------- GRAFICO 3 -------#

# Configuración estética de los gráficos
sns.set_style("whitegrid")

# Contar y calcular porcentajes
conteo_tecnologia = df_max_tecnologia.groupby('TECNOLOGIA_MAX')['CPOB'].nunique()
total_cpob = conteo_tecnologia.sum()
porcentajes = (conteo_tecnologia / total_cpob * 100).round(1)

# Crear el gráfico
plt.figure(figsize=(10, 9))
ax = sns.barplot(x=conteo_tecnologia.index, y=conteo_tecnologia.values, palette=colores)

plt.title('Número de CPOB por Tecnología Predominante', fontsize=16, fontweight='bold')
plt.xlabel('Tecnología Predominante', fontsize=12)
plt.ylabel('Número de CPOB', fontsize=12)
plt.xticks(rotation=0)

# Agregar valores y porcentajes arriba de cada barra
for i, (valor, porcentaje) in enumerate(zip(conteo_tecnologia.values, porcentajes.values)):
    ax.text(i, valor + 0.5, f'{valor}\n({porcentaje}%)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Crear y agregar la leyenda
leyenda = [
    mpatches.Patch(color='red', label='2G = Tecnología mediocre'),
    mpatches.Patch(color='orange', label='3G = Tecnología aceptable'),
    mpatches.Patch(color='green', label='4G = Tecnología buena'),
    mpatches.Patch(color='blue', label='5G = Tecnología excelente')
]
ax.legend(handles=leyenda, title='Leyenda')

plt.tight_layout()
plt.show()


#------- GRAFICO 4 -------#

# Colores por operador
operadores = df_comparativo['OPERADOR_MAX'].unique()
palette = sns.color_palette("Set2", len(operadores))
# `color_dict` moved to top (below imports) to centralize color settings

# Configuración del gráfico
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# Crear gráfico de barras horizontales
ax = sns.barplot(
    x='PORCENTAJE_COBERTURA',
    y='DEPARTAMENTO',
    data=df_comparativo,
    palette=[color_dict[op] for op in df_comparativo['OPERADOR_MAX']]
)

# Etiquetas de valor
for i, v in enumerate(df_comparativo['PORCENTAJE_COBERTURA']):
    ax.text(
        v + (0.5 if v > 0 else 0),  # posición derecha o izquierda
        i,
        f"{abs(v):.1f}%",
        color='black',
        va='center',
        ha='left' if v > 0 else 'right',
        fontsize=10,
        fontweight='bold'
    )

# Línea central en 0
plt.axvline(0, color='black', linewidth=1)

# Títulos y etiquetas
plt.title('Departamentos con mayor y menor cobertura móvil promedio (2024)', fontsize=16, fontweight='bold')
plt.xlabel('Porcentaje de cobertura promedio (%)', fontsize=13)
plt.ylabel('Departamento', fontsize=13)

# Leyenda de operadores
handles = [plt.Rectangle((0,0),1,1, color=color_dict[op]) for op in operadores]
plt.legend(handles, operadores, title='Operador predominante', loc='upper right')

plt.tight_layout()
plt.show()

#------- GRAFICO 5 -------#

# Graficar pastel
fig, ax = plt.subplots(figsize=(8,8))
wedges, texts = ax.pie(
    porcentaje_operador,
    labels=None,  # no mostramos etiquetas directamente
    colors=[color_dict[op] for op in conteo_operador.index],
    startangle=90,
    counterclock=False,
    wedgeprops={'edgecolor':'white', 'linewidth':1.5}
)

# Añadir etiquetas fuera con porcentaje y conteo
for i, w in enumerate(wedges):
    ang = (w.theta2 + w.theta1)/2.  # ángulo medio de la porción
    x = 1.19 * np.cos(np.deg2rad(ang))  # coordenada x
    y = 1.13 * np.sin(np.deg2rad(ang))  # coordenada y
    ax.text(
        x, y,
        f"{conteo_operador.index[i]}\n{porcentaje_operador.values[i]:.1f}%\n({conteo_operador.values[i]})",
        ha='center', va='center', fontsize=11, fontweight='bold'
    )

plt.title('Porcentaje de predominancia por operador (CPOB)', fontsize=16, fontweight='bold')
plt.show()

#------- GRAFICO 6 -------#
