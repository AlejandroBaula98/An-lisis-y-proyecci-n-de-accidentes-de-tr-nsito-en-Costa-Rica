##Proyecto final

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Carga de datos y creación de dataframes
ruta_accidentes = r"C:\Users\Luis\Documents\Ingeniería Ciencia de datos\Segundo cuatrimestre\Introducción a la ciencia de datos\Proyecto grupal\accidentes_2020_2022_limpio.csv"
ruta_personas   = r"C:\Users\Luis\Documents\Ingeniería Ciencia de datos\Segundo cuatrimestre\Introducción a la ciencia de datos\Proyecto grupal\personas_2020_2022_limpio.csv"

df_accidentes = pd.read_csv(ruta_accidentes, encoding="latin-1")
df_personas   = pd.read_csv(ruta_personas, encoding="latin-1")

#Para limpiar errores de formato en columnas que pueden encontrarse en los archivos
df_accidentes.columns = df_accidentes.columns.str.replace("ï»¿", "", regex=False)
df_personas.columns   = df_personas.columns.str.replace("ï»¿", "", regex=False)

#Se identificó un valor en la columna objetivo con NA, por lo que se desfiltra esa fila del dataframe
df_accidentes = df_accidentes[df_accidentes["accidentes_totales"].notna()]

#Desfiltrar la categoría total para que no incluya en el cálculo de las variables
df_accidentes = df_accidentes[df_accidentes["categoria"].str.lower() != "total"]

#Aplicamos una variable "y" como objetivo de predicción
y = df_accidentes["accidentes_totales"]

#"X" serían todo el resto de columnas menos la que definimos como objetivo de predicción
X = df_accidentes.drop(columns=["accidentes_totales"])

X = pd.get_dummies(X, drop_first=True) #Esto convierte las variables categóricas en númericas para que sean interpretadas por el modelo

#Definir el entrenamiento del modelo y la prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #se aplica un porcentaje de 20% de entrenamiento

#Entrenamiento del modelo:
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

#Predicciones para evaluación
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

promedio = df_accidentes["accidentes_totales"].mean() #Determinar la media para el porcentaje de error relativo

error_relativo = (rmse / promedio) * 100

print("Promedio:", promedio)
print("Error relativo:", error_relativo)

#Los resultados obtenidos del primer entrenamiento se presentan a continuación:
#RMSE: 113.62530952487056
#R²: 0.9983770127509908
#Promedio: 1678.5855513307986
#Error relativo: 6.76911042364134

#El RMSE es la raíz del error cuadrático medio, mide qué tan lejos están los valores de las predicciones con respecto a los valores reales
#Es el tamaño promedio del error del modelo, y se expresa en las mismas unidades que la variable objetivo.
#Según el resultado, esto significa que el modelo se equivoca en promedio 113 accidentes con respecto a la realidad de los datos

#El R² es el coeficiente de determinación, y mide la eficiencia del modelo para explicar la variabilidad de los datos reales
#Con un R² de 0.9983, esto significa que el modelo explica el 99.83% de la variación real de los accidentes.
#El modelo entiende los patrones de la información presentada y los reproduce con presición.
#Con este valor se puede decir que el modelo es altamente confiable

#El error relativo indica que existe un porcentaje promedio de error del 6.77% en las predicciones del modelo
#Se puede concluir que el porcentaje de error es relativamente bajo y por tanto también confirma la confianza en el modelo de predicción.

#Crear el df proyectado:
df_proyectado = df_accidentes.copy()

df_proyectado = df_proyectado.drop(columns=["accidentes_totales"])

df_proyectado_2023 = df_proyectado.copy()
df_proyectado_2023["anio"] = 2023

df_proyectado_2024 = df_proyectado.copy()
df_proyectado_2024["anio"] = 2024

df_proyectado_2025 = df_proyectado.copy()
df_proyectado_2025["anio"] = 2025

df_proyectado_total = pd.concat([df_proyectado_2023, df_proyectado_2024, df_proyectado_2025], ignore_index=True)

df_proyectado_codificado = pd.get_dummies(df_proyectado_total, drop_first=True)#Para convertir nuevamente las variables categóricas en númericas

df_proyectado_codificado = df_proyectado_codificado.reindex(columns=X.columns, fill_value=0) #Esto hace que el dataframe proyectado tenga las mismas columnas y orden que el dataframe de entrenamiento "X"

#Aplicar las predicciones
predicciones = model.predict(df_proyectado_codificado)

df_proyectado_total["prediccion_accidentes"] = predicciones

#Resumen para prueba
resumen = df_proyectado_total.groupby("anio")["prediccion_accidentes"].sum()
print(resumen)

#El resultado muestra que hay un error en la aplicación del modelo. El contenido de los datos base está dividido en tablas que tienen subtotales, no se ha hecho un tratamiento adecuado de datos para que el modelo no los interprete como agregación de valores.
#Se realizará el entrenamiento de nuevo a través de una sola dimensión para evaluar los resultados. La dimensión escogida será la tabla de mes presenta una distribución lógica y entendible del universo de los datos.

df_mes = df_accidentes[df_accidentes["tabla"] == "mes"].copy()
df_mes = df_mes[df_mes["categoria"].str.lower() != "total"]

y = df_mes["accidentes_totales"]

X = df_mes.drop(columns=["accidentes_totales"])

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred_mes = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_mes)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_mes)

print("Resultados con filtro por dimensión -mes-")
print("RMSE:", rmse)
print("R²:", r2)

promedio = df_mes["accidentes_totales"].mean()

error_relativo = (rmse / promedio) * 100

print("Promedio:", promedio)
print("Error relativo:", error_relativo)

#Nuevos resultados obtenidos por el entrenamiento del modelo usando la dimensión mes:
#RMSE: 33.31855024794748
#R²: 0.8830998547140597
#Promedio: 1149.8333333333333
#Error relativo: 2.8976851933277996

#Con la nueva evaluación, los resultados aún mantienen una representación estable.

#Creación de los dataframe de proyecciones:
df_futuro_base = df_mes.drop(columns=["accidentes_totales"]).copy()
if "tabla" in df_futuro_base.columns:
    df_futuro_base = df_futuro_base.drop(columns=["tabla"])

dfs_futuros = []

for anio_futuro in [2023, 2024, 2025]:
    tmp = df_futuro_base.copy()
    tmp["anio"] = anio_futuro          
    tmp["anio_pred"] = anio_futuro  
    dfs_futuros.append(tmp)

df_proyectado_total = pd.concat(dfs_futuros, ignore_index=True)

df_proyectado_codificado = pd.get_dummies(
    df_proyectado_total.drop(columns=["anio_pred"]),
    drop_first=True
)

df_proyectado_codificado = df_proyectado_codificado.reindex(
    columns=X.columns,
    fill_value=0
)

# Predicciones
predicciones = model.predict(df_proyectado_codificado)
df_proyectado_total["prediccion_accidentes"] = predicciones

# Resumen para las predicciones
resumen = df_proyectado_total.groupby("anio_pred")["prediccion_accidentes"].sum()
print("\nPredicción total por año:")
print(resumen)

#Resultados de predicción:
#2023    41251.355
#2024    41251.355
#2025    41251.355

#El modelo no está identificando los años como una tendencia temporal. Se aplicarán ajustes adicionales para afinar el entrenamiento.
#Identificando el funcionamiento del algoritmo, el modelo de random forest no genera predicción temporal para el caso de los datos usados.
#El modelo sirve para identificar patrones de distribución y relaciones, pero no aplica un entendimiento correcto en aspectos de crecimiento o decrecimiento por año
#Se aplicará un modelo de regresión lineal combinado con Random Forest para tener proyecciones anuales complementadas con la distribución de variables de los árboles de decisión.

df_anual = df_mes.groupby("anio")["accidentes_totales"].sum().reset_index()
print("\nTotales anuales observados del 2020 al 2022):")
print(df_anual)

#Creación de variables para regresión lineal y entrenamiento del modelo
X_anual = df_anual[["anio"]]               
y_anual = df_anual["accidentes_totales"]

model_lineal = LinearRegression()
model_lineal.fit(X_anual, y_anual)

anios_futuros = pd.DataFrame({"anio": [2023, 2024, 2025]})

totales_futuros = model_lineal.predict(anios_futuros)

print("\nTotales anuales proyectados por regresión lineal:")

anios = list(anios_futuros["anio"])
totales = list(totales_futuros)

for i in range(len(anios)):
    anio = anios[i]
    total = totales[i]
    print(f"{anio}: {total:.2f}")

#Resultado de años proyectados:
#2023: 17420.00
#2024: 19231.00
#2025: 21042.00

#Los resultados anteriores tienen más sentido en relación con el comportamiento por año usado de base.
#Ahora aplicamos el random forest por mes:

df_futuro_base = df_mes.drop(columns=["accidentes_totales"]).copy()

dfs_futuros = []

for anio_futuro in [2023, 2024, 2025]:
    tmp = df_futuro_base.copy()
    tmp["anio"] = anio_futuro          #tmp se usa como variable temporal
    tmp["anio_pred"] = anio_futuro     
    dfs_futuros.append(tmp)

df_proyectado_mensual = pd.concat(dfs_futuros, ignore_index=True)

df_proyectado_codificado = pd.get_dummies(
    df_proyectado_mensual.drop(columns=["anio_pred"]),
    drop_first=True
)

df_proyectado_codificado = df_proyectado_codificado.reindex( #Para alinear las columnas
    columns=X.columns,
    fill_value=0
)

pred_rf = model.predict(df_proyectado_codificado)
df_proyectado_mensual["pred_rf"] = pred_rf

#Juntar los resultados del random forest con respecto a los montos proyectados de la regresión lineal:
#Suma de las predicciones del random forest:
tot_rf_por_anio = df_proyectado_mensual.groupby("anio_pred")["pred_rf"].sum().reset_index()
tot_rf_por_anio = tot_rf_por_anio.rename(columns={"pred_rf": "total_rf"})

#Totales anuales de la regresión lineal:
df_tot_lin = anios_futuros.copy()
df_tot_lin["total_lin"] = totales_futuros

# Unir ambas tablas por año
df_unificado = tot_rf_por_anio.merge(
    df_tot_lin,
    left_on="anio_pred",
    right_on="anio",
    how="left"
)

df_unificado["factor_escala"] = df_unificado["total_lin"] / df_unificado["total_rf"]

print("\nFactores de escala por año (para ajustar random forest a regresión lineal):")
print(df_unificado[["anio_pred", "total_rf", "total_lin", "factor_escala"]])

df_proyectado_mensual = df_proyectado_mensual.merge(
    df_unificado[["anio_pred", "factor_escala"]],
    on="anio_pred",
    how="left"
)

df_proyectado_mensual["pred_final"] = df_proyectado_mensual["pred_rf"] * df_proyectado_mensual["factor_escala"]

resumen_combinacion = df_proyectado_mensual.groupby("anio_pred")["pred_final"].sum()
print("\nTotales anuales después del ajuste de ambos modelos")
print(resumen_combinacion)

orden_meses = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "setiembre", "octubre", "noviembre", "diciembre"
]

df_mes["categoria"] = pd.Categorical(df_mes["categoria"],
                                     categories=orden_meses,
                                     ordered=True)

df_proyectado_mensual["categoria"] = pd.Categorical(df_proyectado_mensual["categoria"],
                                                    categories=orden_meses,
                                                    ordered=True)

resumen_mensual = df_proyectado_mensual.groupby(["anio_pred", "categoria"])["pred_final"].sum().reset_index()
print("\nPredicción combinada por año y mes:")
print(resumen_mensual)

#Se aplica un factor de escala para convertir los valores del random forest a el valor proyectado por la regresión lineal:
#0.422289 para 2023
#0.466191 para 2024
#0.510092 para 2025
#Lo anterior nos muestra una distribución proyectada de eventos por mes y año

#Gráfico para ver la comparación entre los datos base y los proyectados:
df_hist_plot = df_mes.copy()
df_hist_plot = df_hist_plot[df_hist_plot["categoria"].str.lower() != "total"]

df_hist_plot = df_hist_plot[["anio", "categoria", "accidentes_totales"]].copy()
df_hist_plot = df_hist_plot.rename(columns={"accidentes_totales": "valor"})

df_proj_plot = resumen_mensual.rename(columns={
    "anio_pred": "anio",
    "pred_final": "valor"
})[["anio", "categoria", "valor"]].copy()

df_grafico = pd.concat([df_hist_plot, df_proj_plot], ignore_index=True)

df_grafico["categoria"] = df_grafico["categoria"].str.lower()
df_grafico = df_grafico[df_grafico["categoria"].isin(orden_meses)].copy()

df_grafico["categoria"] = pd.Categorical(
    df_grafico["categoria"],
    categories=orden_meses,
    ordered=True
)

df_grafico["anio"] = df_grafico["anio"].astype(int)

plt.figure(figsize=(10, 6))

for anio in sorted(df_grafico["anio"].unique()):
    df_anio = df_grafico[df_grafico["anio"] == anio].sort_values("categoria")
    plt.plot(df_anio["categoria"], df_anio["valor"],
             marker="o", label=str(anio))

plt.title("Accidentes por mes y año (histórico y proyectado)")
plt.xlabel("Mes")
plt.ylabel("Cantidad de accidentes")
plt.xticks(rotation=45)
plt.legend(title="Año")
plt.grid(True)

plt.tight_layout()
plt.show()

#Observación final: Según el gráfico, la proyección de cada año es similar en su comportamiento de crecimiento y decrecimiento mensual.
#Esto en apariencia indica que los datos base no son lo suficientemente extensos para que el modelo interprete la variación anual como algo significativo.
#Adicionalmente, puede ser causa de como se configuró el modelo, usando la tabla de mes como registro para la predicción, pero en una estructura agrupada.
#El modelo de random Forest entrenado con la dimensión mes aprendió patrones que se presentan a través de esos meses y los utilizó para proyectar esos patrones.
#Parece que no hay suficiente variabilidad estadística con una muestra de 3 años para que el modelo reconozca un patrón anual
