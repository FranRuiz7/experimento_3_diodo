import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =============================================================================
# 1. PARÁMETROS EXPERIMENTALES Y CARGA TRAZABLE DE DATOS CRUDOS
# =============================================================================
R_med = 985.0       # Resistencia medida en Ohmios (0.985 kOhm)
u_R = 11.0          # Incertidumbre estándar GUM en Ohmios (0.011 kOhm)
archivo_csv = "parteA_curva_IV.csv"

# Lectura adaptable al formato de exportación de PASCO Capstone (coma o punto)
df_raw = pd.read_csv(archivo_csv, sep=None, engine="python", decimal=",")
if df_raw.shape[1] < 3 or df_raw.dtypes.iloc[0] == "object":
    df_raw = pd.read_csv(archivo_csv, sep=None, engine="python", decimal=".")

# Nos quedamos con las 3 primeras columnas (Tiempo, Vs, VD) y limpiamos filas vacías
df = df_raw.iloc[:, :3].copy()
df.columns = ["t", "Vs", "VD"]
df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# =============================================================================
# 2. CÁLCULO DE VARIABLES INDIRECTAS (SIN ORDENAR POR TENSIÓN)
# =============================================================================
# Ecuación (9) de la guía: VR(t) = Vs(t) - VD(t) e ID(t) = VR(t) / R
df["VR"] = df["Vs"] - df["VD"]
df["ID"] = df["VR"] / R_med

# =============================================================================
# 3. IDENTIFICACIÓN DE CICLOS Y SEPARACIÓN DE RAMAS (ASCENDENTE vs DESCENDENTE)
# =============================================================================
# A fs = 2000 Hz y f = 1 Hz, cada ciclo tiene ~2000 muestras.
# Buscamos los picos (máximos de ~3.5 V) y valles (mínimos de ~-3.5 V) de Vs(t)
idx_picos, _ = find_peaks(df["Vs"], distance=1500, prominence=5.0)
idx_valles, _ = find_peaks(-df["Vs"], distance=1500, prominence=5.0)

# Clasificamos cada punto temporal según si la rampa sube o baja
# Usamos la derivada suavizada de Vs respecto del tiempo para no amplificar ruido
dVs_dt = np.gradient(df["Vs"].rolling(window=21, center=True, min_periods=1).mean(), df["t"])
df["rama"] = np.where(dVs_dt >= 0, "Ascendente", "Descendente")

# Separamos los subconjuntos para comparar ambas ramas
df_asc = df[df["rama"] == "Ascendente"].copy()
df_desc = df[df["rama"] == "Descendente"].copy()

# Verificación cuantitativa de superposición (interpolando en una grilla común de VD)
vd_grilla = np.linspace(-3.0, 0.63, 200)
# Ordenamos copias auxiliares SOLO después de haber separado las ramas
asc_sort = df_asc.sort_values("VD")
desc_sort = df_desc.sort_values("VD")
id_asc_interp = np.interp(vd_grilla, asc_sort["VD"], asc_sort["ID"])
id_desc_interp = np.interp(vd_grilla, desc_sort["VD"], desc_sort["ID"])
dif_max_ramas = np.max(np.abs(id_asc_interp - id_desc_interp))

# =============================================================================
# 4. ANÁLISIS DEL PISO DE RUIDO Y CRITERIO OPERACIONAL DE TENSIÓN UMBRAL
# =============================================================================
# En polarización inversa franca (VD < -0.5 V), la corriente ideal es ~0 A
ruido_inversa = df[df["VD"] < -0.5]["ID"]
media_ruido = ruido_inversa.mean()
std_ruido = ruido_inversa.std()

# Umbral conservador para el logaritmo: corrientes 5 veces por encima del desvío del ruido
# y estrictamente mayores a 10 uA (0.00001 A) para estar fuera del piso instrumental
umbral_ruido_ID = max(abs(media_ruido) + 5 * std_ruido, 1e-5)

# Criterio operacional para la "tensión umbral": tensión VD cuando ID = 1.0 mA (0.001 A)
ID_criterio = 1.0e-3  # 1 mA
ventana_criterio = df[(df["ID"] >= 0.95e-3) & (df["ID"] <= 1.05e-3)]
VD_umbral_op = ventana_criterio["VD"].mean()
u_VD_umbral_op = ventana_criterio["VD"].std()

# =============================================================================
# 5. FILTRADO PARA CONDUCCIÓN DIRECTA Y ESCALA SEMILOGARÍTMICA
# =============================================================================
df_directa = df[(df["VD"] > 0.2) & (df["ID"] > umbral_ruido_ID)].copy()
df_directa["ln_ID"] = np.log(df_directa["ID"])

# =============================================================================
# 6. REPORTE EN CONSOLA Y GENERACIÓN DE GRÁFICAS DE LA SECCIÓN 6.1
# =============================================================================
print("=== RESUMEN SECCIÓN 6.1 ===")
print(f"Total de muestras cargadas: {len(df)} (Tiempo: {df['t'].min():.2f} s a {df['t'].max():.2f} s)")
print(f"Picos detectados en Vs(t): {len(idx_picos)} | Valles detectados: {len(idx_valles)}")
print(f"Corriente media en corte (VD < -0.5 V): {media_ruido*1e6:.2f} uA (Ruido std: {std_ruido*1e6:.2f} uA)")
print(f"Diferencia máxima entre rama ascendente y descendente: {dif_max_ramas*1e6:.2f} uA")
print(f"Umbral mínimo de corriente para escala semilog: {umbral_ruido_ID*1e6:.2f} uA")
print(f"Tensión umbral operacional (a ID = 1.0 mA): VD = ({VD_umbral_op:.4f} +/- {u_VD_umbral_op:.4f}) V")

# --- FIGURA 1: Comparación de ramas Ascendente y Descendente en escala lineal ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(df_asc["VD"], df_asc["ID"] * 1e3, ".", markersize=2, alpha=0.4, label="Rama Ascendente", color="tab:blue")
ax1.plot(df_desc["VD"], df_desc["ID"] * 1e3, ".", markersize=2, alpha=0.4, label="Rama Descendente", color="tab:orange")
ax1.axhline(ID_criterio * 1e3, color="gray", linestyle="--", linewidth=1, label=f"Criterio $I_D$ = 1 mA ($V_D \\approx$ {VD_umbral_op:.2f} V)")
ax1.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax1.set_ylabel("Corriente en el diodo $I_D$ (mA)")
ax1.set_title("Curva característica $I_D(V_D)$ - Escala Lineal")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# --- FIGURA 2: Conducción directa en escala semilogarítmica ---
df_dir_asc = df_directa[df_directa["rama"] == "Ascendente"]
df_dir_desc = df_directa[df_directa["rama"] == "Descendente"]

ax2.plot(df_dir_asc["VD"], df_dir_asc["ln_ID"], ".", markersize=3, alpha=0.5, label="Ascendente ($I_D > I_{ruido}$)", color="tab:blue")
ax2.plot(df_dir_desc["VD"], df_dir_desc["ln_ID"], ".", markersize=3, alpha=0.5, label="Descendente ($I_D > I_{ruido}$)", color="tab:orange")
ax2.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax2.set_ylabel("$\\ln(I_D / \\mathrm{1\\,A})$")
ax2.set_title("Región de Conducción Directa - Escala Semilogarítmica")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("seccion_6_1_curvas_IV.png", dpi=300)
plt.show()