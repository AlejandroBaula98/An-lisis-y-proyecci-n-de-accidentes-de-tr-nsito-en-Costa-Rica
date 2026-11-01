import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Carga de datos-
ruta_accidentes = r"C:\Users\Luis\Documents\Ingeniería Ciencia de datos\Segundo cuatrimestre\Introducción a la ciencia de datos\Proyecto grupal\accidentes_2020_2022_limpio.csv"
ruta_personas   = r"C:\Users\Luis\Documents\Ingeniería Ciencia de datos\Segundo cuatrimestre\Introducción a la ciencia de datos\Proyecto grupal\personas_2020_2022_limpio.csv"

df_accidentes = pd.read_csv(ruta_accidentes, encoding="latin-1")
df_personas   = pd.read_csv(ruta_personas, encoding="latin-1")

# Arreglo de formato en primera columna
df_accidentes.columns = df_accidentes.columns.str.replace("ï»¿", "", regex=False)
df_personas.columns   = df_personas.columns.str.replace("ï»¿", "", regex=False)

# Estructura

print("Estructura base datos: Accidentes")
df_accidentes.info()
print(f"\nShape accidentes → Filas: {df_accidentes.shape[0]}, Columnas: {df_accidentes.shape[1]}")
print("\nColumnas accidentes:")
print(df_accidentes.columns.tolist())

print("\n\nEstructura base de datos: Personas")
df_personas.info()
print(f"\nShape personas   → Filas: {df_personas.shape[0]}, Columnas: {df_personas.shape[1]}")
print("\nColumnas personas:")
print(df_personas.columns.tolist())

#Quitar la categoría "Totales"
def quitar_totales(df):
    if "categoria" in df.columns:
        return df[df["categoria"].str.lower() != "total"].copy()
    return df.copy()

df_acc_sin_total = quitar_totales(df_accidentes)
df_per_sin_total = quitar_totales(df_personas)


#Resumen estádistico
print("\n\nResumen estadístico base de datos: Accidentes")
desc_acc = df_acc_sin_total.select_dtypes(include=["number"]).describe().round(2)
print(desc_acc.transpose())   # transpuesto para que cada variable sea una fila

print("\n\nResumen estadístico base de datos: Personas")
desc_per = df_per_sin_total.select_dtypes(include=["number"]).describe().round(2)
print(desc_per.transpose())


#Totales por año
print("\n\nTotales de accidentes por año")
tot_acc_anio = (
    df_accidentes[df_accidentes["categoria"].str.lower() == "total"]
    .groupby("anio")["accidentes_totales"]
    .max()
)
print(tot_acc_anio)

print("\n\nTotales base de datos de Personas por año")
tot_per_anio = (
    df_personas[df_personas["categoria"].str.lower() == "total"]
    .groupby("anio")["personas_totales"]
    .max()
)
print(tot_per_anio)

#Promedio de accidentes
print("\n\nPromedio de accidentes por dimensión")
prom_acc_por_tabla = (
    df_acc_sin_total
    .groupby("tabla")["accidentes_totales"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)
print(prom_acc_por_tabla)

print("\n\nPromedio de personas por dimensión")
prom_per_por_tabla = (
    df_per_sin_total
    .groupby("tabla")["personas_totales"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)
print(prom_per_por_tabla)


#Promedio por tabla y categoría
prom_per_por_tabla_cat = (
    df_per_sin_total
    .groupby(["tabla", "categoria"])["personas_totales"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print("\n\nPromedio de personas por tabla y categoría")
print(prom_per_por_tabla_cat)

#Construcción de visualizaciones
df_totales = df_accidentes[df_accidentes["categoria"].str.lower() == "total"].copy()

acc_por_anio = (
    df_totales
    .groupby("anio")["accidentes_totales"]
    .max()
)

#Primer gráfico
plt.figure(figsize=(10, 6))
bars = plt.bar(acc_por_anio.index.astype(str), acc_por_anio.values)

plt.title("Accidentes por Año", fontsize=16)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Total de Accidentes", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",   #separador de miles
        ha='center',
        va='bottom',
        fontsize=10
    )
plt.show()


# Gráfico por tipos de accidente
df_tipo = df_accidentes[
    (df_accidentes["categoria"].str.lower() != "total") &
    (df_accidentes["tabla"] == "tipo_accidente")
].copy()

acc_por_tipo = (
    df_tipo.groupby("categoria")["accidentes_totales"]
    .sum()
    .sort_values(ascending=True)
)

#Segundo gráfico
N = 8
s = acc_por_tipo.sort_values(ascending=False)

top = s.head(N)
otros = s.iloc[N:].sum()

plot_s = top.copy()
if otros > 0:
    plot_s.loc["otros (agrupado)"] = otros

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.barh(plot_s.index[::-1], plot_s.values[::-1])

ax.set_title("Tipo de accidente", fontsize=22, pad=12)
ax.set_xlabel("Total de accidentes", fontsize=20)
ax.set_ylabel("Tipo de accidente", fontsize=20)

ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)

ax.grid(True, axis="x", linestyle="--", alpha=0.35)

max_val = plot_s.max()
ax.set_xlim(0, max_val * 1.12)

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + max_val * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{int(w):,}",
        va="center",
        fontsize=16
    )

fig.tight_layout()
fig.savefig("barras_tipo_accidente.png", dpi=300, bbox_inches="tight")
plt.show()

#Heatmap de accidentes por provincia
df_prov = df_accidentes[
    (df_accidentes["categoria"].str.lower() != "total") &
    (df_accidentes["tabla"] == "provincia")
].copy()

# 2) Crear tabla pivot
pivot = df_prov.pivot_table(
    index="categoria",
    columns="anio",
    values="accidentes_totales",
    aggfunc="sum"
).sort_index()

#Rellenar NaN y convertir a int
pivot = pivot.fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(12, 8))

hm = sns.heatmap(
    pivot.T,            
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"size": 22},
    ax=ax
)

ax.set_title("Mapa de Calor: Accidentes por Provincia y Año", fontsize=22)
ax.set_xlabel("Provincia", fontsize=20)
ax.set_ylabel("Año", fontsize=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=21)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label("",fontsize=20)

plt.tight_layout()

fig.savefig(
    "heatmap_provincias.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show() 

# Histograma de accidentes por hora del día
df_hora = df_accidentes[
    (df_accidentes["tabla"] == "hora") &
    (df_accidentes["categoria"].str.lower() != "total")
].copy()

df_hora["hora_num"] = (
    df_hora["categoria"]
    .str.extract(r"^(\d{2})")[0]
    .astype(int)
)

df_hora = df_hora.sort_values("hora_num")

plt.figure(figsize=(14, 8), constrained_layout=True)

sns.barplot(
    data=df_hora,
    x="hora_num",
    y="accidentes_totales",
    errorbar=None
)

plt.title("Accidentes por Hora del Día", fontsize=16)
plt.xlabel("Hora del Día", fontsize=12)
plt.ylabel("Total de Accidentes", fontsize=12)

plt.tight_layout()
plt.show()

#Gráfico por condiciones climáticas
df_clima = df_accidentes[
    (df_accidentes["tabla"] == "clima") &
    (df_accidentes["categoria"].str.lower() != "total")
].copy()

#Agrupar por condición climática y sumar accidentes
acc_por_clima = (
    df_clima
    .groupby("categoria")["accidentes_totales"]
    .sum()
    .reset_index()
)

acc_por_clima = acc_por_clima.sort_values("accidentes_totales", ascending=False)

#Crear gráfico
fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(
    data=acc_por_clima,
    x="categoria",
    y="accidentes_totales",
    errorbar=None,
    ax=ax
)

ax.set_title("Accidentes por Condición Climática", fontsize=22, pad=12)
ax.set_xlabel("Condición Climática", fontsize=18)
ax.set_ylabel("Total de Accidentes (logarítmico)", fontsize=18)

ax.tick_params(axis="x", labelsize=16, rotation=25)
ax.tick_params(axis="y", labelsize=16)

ax.set_yscale("log")

for bar in ax.patches:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h * 1.08,                 
        f"{int(h):,}",
        ha="center",
        va="bottom",
        fontsize=16
    )

ax.grid(True, axis="y", linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig("barplot_clima_log.png", dpi=300, bbox_inches="tight")
plt.show()

#Pie Chart personas por edad
df_sexo = df_personas[
    (df_personas["tabla"] == "sexo") &
    (df_personas["categoria"].str.lower() != "total")
].copy()

per_por_sexo = (
    df_sexo.groupby("categoria")["personas_totales"]
    .sum()
    .sort_values(ascending=False)
)

#Crear pie chart
fig, ax = plt.subplots(figsize=(7, 5))

labels = per_por_sexo.index.tolist()
values = per_por_sexo.values

wedges, _, autotexts = ax.pie(
    values,
    labels=None,              
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.70,         
    radius=0.95,
    textprops={"fontsize": 20}
)

for t in autotexts:
    t.set_fontsize(18)

ax.set_title("Accidentes por Género", fontsize=22, pad=10)
ax.axis("equal")

ax.legend(
    wedges, labels,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=18,
    title="Género",
    title_fontsize=18,
    frameon=False
)

fig.tight_layout()

fig.savefig("piechart_genero.png", dpi=300, bbox_inches="tight")
plt.show()

#Gráfico de accidentes por edad
df_edad = df_personas[
    (df_personas["tabla"] == "edad") &
    (df_personas["categoria"].str.lower() != "total")
].copy()

df_edad = df_edad[df_edad["categoria"].astype(str).str.fullmatch(r"\d+")]

df_edad["edad"] = df_edad["categoria"].astype(int)

per_por_edad = (
    df_edad.groupby("edad", as_index=False)["personas_totales"]
    .sum()
    .sort_values("edad")
)

bins = [0, 12, 17, 24, 34, 44, 54, 64, 74, 120]
orden_rangos = ["0–12", "13–17", "18–24", "25–34", "35–44",
                "45–54", "55–64", "65–74", "75+"]

per_por_edad["rango_edad"] = pd.cut(
    per_por_edad["edad"],
    bins=bins,
    labels=orden_rangos,
    include_lowest=True,
    right=True
)

edad_binned = (
    per_por_edad.groupby("rango_edad", observed=True, as_index=False)["personas_totales"]
    .sum()
)

edad_binned["rango_edad"] = pd.Categorical(
    edad_binned["rango_edad"],
    categories=orden_rangos,
    ordered=True
)
edad_binned = edad_binned.sort_values("rango_edad")

fig, ax = plt.subplots(figsize=(11, 6))

sns.lineplot(
    data=edad_binned,
    x="rango_edad",
    y="personas_totales",
    marker="o",
    linewidth=2.5,
    ax=ax
)

ax.set_title("Personas involucradas en accidentes por rango de edad", fontsize=22, pad=12)
ax.set_xlabel("Rango de edad", fontsize=18)
ax.set_ylabel("Total de personas", fontsize=18)

ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)

ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()

fig.savefig(
    "personas_por_edad.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Identificar valores aislados
df_edad = df_personas[
    (df_personas["tabla"] == "edad") &
    (df_personas["categoria"].str.lower() != "total")
].copy()

df_edad = df_edad[df_edad["categoria"].str.fullmatch(r"\d+")]

df_edad["edad"] = df_edad["categoria"].astype(int)

#Calcular el Z-Score
media = df_edad["edad"].mean()
std = df_edad["edad"].std(ddof=0)
df_edad["zscore"] = (df_edad["edad"] - media) / std

#Outliers se categorizan como Z > 3
outliers = df_edad[np.abs(df_edad["zscore"]) > 3]

print("Estadísticas de Edad")
print(f"Media: {media:.2f}")
print(f"Desviación Estándar: {std:.2f}\n")

print("Valores Atípicos (Z-Score > 3)")
if outliers.empty:
    print("No se encontraron valores atípicos según el criterio de Z-Score.")
else:
    print(outliers[["edad", "personas_totales", "zscore"]])

#Boxplot accidentes por tipo de vehiculo
df_veh = df_personas[
    df_personas["tabla"].isin(["vehiculo", "categoria_corregida"])
].copy()

df_veh = df_veh[df_veh["categoria"].str.lower() != "total"]

def agrupar_vehiculo(cat):
    cat = cat.lower()
    
    if cat in ["automovil"]:
        return "automóvil"
    elif cat in ["motocicleta"]:
        return "motocicleta"
    elif cat in ["bicicleta"]:
        return "bicicleta"
    elif cat in ["autobus", "microbus", "taxi"]:
        return "transporte público"
    elif cat in [
        "camion", "pick up",
        "cabezal sin remolque",
        "cabezal con remolque",
        "vehiculo con remolque"
    ]:
        return "vehículos de carga"
    elif cat in ["ambulancia", "grua"]:
        return "emergencia / especial"
    elif cat == "rural":
        return "rural"
    else:
        return "otros"

df_veh["tipo_vehiculo_agrupado"] = df_veh["categoria"].apply(agrupar_vehiculo)

orden = (
    df_veh
    .groupby("tipo_vehiculo_agrupado")["personas_totales"]
    .median()
    .sort_values()
    .index
)

plt.close("all")
fig, ax = plt.subplots(figsize=(11, 6))

sns.boxplot(
    data=df_veh,
    x="tipo_vehiculo_agrupado",
    y="personas_totales",
    order=orden,
    showfliers=True,
    ax=ax
)

ax.set_title(
    "Distribución de personas involucradas por tipo de vehículo",
    fontsize=22,
    pad=12
)
ax.set_xlabel("Tipo de vehículo", fontsize=20)
ax.set_ylabel("Personas totales", fontsize=20)

ax.tick_params(axis="x", labelsize=18, rotation=35)
ax.tick_params(axis="y", labelsize=18)

ax.grid(True, axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

fig.savefig(
    "boxplot_personas_por_vehiculo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#Resumen descriptivo general
desc = df_veh["personas_totales"].describe()
print("\nResumen Descriptivo de Personas por tipo de Vehículo")
print(desc)

#Cálculo de IQR y límites
q1 = desc["25%"]
q3 = desc["75%"]
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("\nIQR y límites")
print(f"Q1 (25%): {q1:.2f}")
print(f"Q3 (75%): {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Límite inferior: {lower:.2f}")
print(f"Límite superior: {upper:.2f}")

#Detectar outliers según IQR
outliers = df_veh[
    (df_veh["personas_totales"] < lower) |
    (df_veh["personas_totales"] > upper)
]

#Resumen por tipo de vehículo
print("\nResumen por tipo de vehículo")
resumen_por_vehiculo = (
    df_veh
    .groupby("categoria")["personas_totales"]
    .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(2)
)
print(resumen_por_vehiculo.to_string())

# Relaciones entre variables
df_hora = df_accidentes[df_accidentes["tabla"] == "hora"].copy()

# Extraer hora solo si empieza con dos dígitos
df_hora["hora_extraida"] = df_hora["categoria"].str.extract(r"^(\d{2})")

# Eliminar filas donde no se pudo extraer una hora válida
df_hora = df_hora.dropna(subset=["hora_extraida"])

# Convertir a entero
df_hora["hora_num"] = df_hora["hora_extraida"].astype(int)

# Calcular correlación
correlacion = df_hora["hora_num"].corr(df_hora["accidentes_totales"])

print("\nCorrelación Hora vs Accidentes:", correlacion)

# Tipo
if correlacion > 0:
    tipo = "positiva"
elif correlacion < 0:
    tipo = "negativa"
else:
    tipo = "nula"

# Fuerza
abs_corr = abs(correlacion)
if abs_corr > 0.7:
    fuerza = "fuerte"
elif abs_corr > 0.4:
    fuerza = "moderada"
else:
    fuerza = "débil"

print("La correlación es", tipo, "y", fuerza)

# Interpretación causal
print("\nCausalidad:")
if tipo == "positiva":
    print("Al aumentar la hora del día, tienden a aumentar los accidentes (no causal).")
elif tipo == "negativa":
    print("A medida que avanza la hora, tienden a disminuir los accidentes (no causal).")
else:
    print("No hay relación lineal clara entre hora del día y cantidad de accidentes.")

# Gráfico
x = df_hora["hora_num"].to_numpy()
y = df_hora["accidentes_totales"].to_numpy()

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(x, y, alpha=0.7, label="Datos reales")

coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)

x_sorted = np.sort(x)

ax.plot(
    x_sorted,
    poly1d_fn(x_sorted),
    linestyle="--",
    linewidth=2.5,
    color="red",
    label=f"Tendencia: y = {coef[0]:.2f}x + {coef[1]:.2f}"
)

ax.set_title("Relación entre Hora del Día y Accidentes", fontsize=22, pad=12)
ax.set_xlabel("Hora del día (0–23)", fontsize=20)
ax.set_ylabel("Total de accidentes", fontsize=20)

ax.set_xticks(np.arange(0, 24, 1))
ax.tick_params(axis="x", labelsize=18, rotation=0)
ax.tick_params(axis="y", labelsize=18)

ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=14)

plt.tight_layout()

fig.savefig(
    "accidentes_por_hora_scatter_tendencia.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
