import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# 1. CARGA DE DATOS Y CONSTANTES FÍSICAS FIJAS
# =============================================================================
R_med = 985.0        # Resistencia medida en Ohmios (0.985 kOhm)
u_R = 11.0           # Incertidumbre estándar de R en Ohmios
VT = 0.0257          # Voltaje térmico fijo a 25 °C (25.7 mV = 0.0257 V)
archivo_csv = "parteA_curva_IV.csv"

# Lectura adaptable a coma o punto decimal
df_raw = pd.read_csv(archivo_csv, sep=None, engine="python", decimal=",")
if df_raw.shape[1] < 3 or df_raw.dtypes.iloc[0] == "object":
    df_raw = pd.read_csv(archivo_csv, sep=None, engine="python", decimal=".")

df = df_raw.iloc[:, :3].copy()
df.columns = ["t", "Vs", "VD"]
df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# Reconstrucción de VR e ID (Ecuación 9)
df["VR"] = df["Vs"] - df["VD"]
df["ID"] = df["VR"] / R_med

# Filtramos datos con corriente positiva por encima del ruido (> 15 uA) para logaritmo
df_pos = df[(df["VD"] > 0.25) & (df["ID"] > 1.5e-5)].copy()
df_pos["ln_ID"] = np.log(df_pos["ID"])

# =============================================================================
# 2. SELECCIÓN DEL INTERVALO Y ESTIMACIÓN INICIAL SEMILOG (Ecuaciones 4 y 5)
# =============================================================================
# Intervalo principal de ajuste donde ln(ID) vs VD es aproximadamente lineal
VD_min, VD_max = 0.42, 0.62
df_fit = df_pos[(df_pos["VD"] >= VD_min) & (df_pos["VD"] <= VD_max)].copy()

# Ajuste lineal en escala semilogarítmica: ln(ID) = m * VD + b (Ecuación 4)
p_lin, cov_lin = np.polyfit(df_fit["VD"], df_fit["ln_ID"], 1, cov=True)
m_semilog, b_semilog = p_lin
u_m, u_b = np.sqrt(np.diag(cov_lin))

# Despeje de parámetros iniciales (semillas) mediante la Ecuación 5
n_semilog = 1.0 / (m_semilog * VT)
u_n_semilog = (u_m / (m_semilog**2 * VT))
Is_semilog = np.exp(b_semilog)
u_Is_semilog = Is_semilog * u_b

# =============================================================================
# 3. AJUSTE NO LINEAL DIRECTO DEL MODELO DE SHOCKLEY SIMPLE (Ecuación 1)
# =============================================================================
def modelo_shockley(VD, Is, n):
    """Ecuación (1): Modelo ideal de Shockley con VT fijo."""
    return Is * (np.exp(VD / (n * VT)) - 1.0)

# Usamos las estimaciones del método semilog como semillas (p0)
p0_shockley = [Is_semilog, n_semilog]
popt_sh, pcov_sh = curve_fit(
    modelo_shockley, df_fit["VD"], df_fit["ID"], p0=p0_shockley, maxfev=10000
)
Is_nl, n_nl = popt_sh
u_Is_nl, u_n_nl = np.sqrt(np.diag(pcov_sh))

# Cálculo de residuos (Ecuación 13): r_i = ID_i - I_modelo(VD_i)
df_fit["ID_modelo_nl"] = modelo_shockley(df_fit["VD"], Is_nl, n_nl)
df_fit["ID_modelo_semilog"] = modelo_shockley(df_fit["VD"], Is_semilog, n_semilog)
df_fit["residuos_nl"] = df_fit["ID"] - df_fit["ID_modelo_nl"]
df_fit["residuos_semilog"] = df_fit["ID"] - df_fit["ID_modelo_semilog"]

# Evaluamos también los residuos en todo el rango de conducción directa (0.35 V hasta el máximo)
df_eval = df_pos[df_pos["VD"] >= 0.35].copy()
df_eval["residuos_nl"] = df_eval["ID"] - modelo_shockley(df_eval["VD"], Is_nl, n_nl)

# =============================================================================
# 4. ANÁLISIS DE SENSIBILIDAD AL MODIFICAR EL INTERVALO DE AJUSTE (Punto 6.2.5)
# =============================================================================
intervalos_prueba = [
    (0.38, 0.58),
    (0.40, 0.60),
    (0.42, 0.62),  # Intervalo central de referencia
    (0.45, 0.63),
    (0.48, 0.65),
]

resultados_sensibilidad = []
for vmin, vmax in intervalos_prueba:
    sub = df_pos[(df_pos["VD"] >= vmin) & (df_pos["VD"] <= vmax)]
    if len(sub) > 20:
        # Semilog
        m_s, b_s = np.polyfit(sub["VD"], sub["ln_ID"], 1)
        n_s = 1.0 / (m_s * VT)
        Is_s = np.exp(b_s)
        # No lineal directo
        popt_sub, pcov_sub = curve_fit(
            modelo_shockley, sub["VD"], sub["ID"], p0=[Is_s, n_s], maxfev=10000
        )
        err_sub = np.sqrt(np.diag(pcov_sub))
        rms_res = np.sqrt(np.mean((sub["ID"] - modelo_shockley(sub["VD"], *popt_sub))**2))
        resultados_sensibilidad.append({
            "Intervalo VD (V)": f"[{vmin:.2f}, {vmax:.2f}]",
            "N puntos": len(sub),
            "Is_semilog (nA)": Is_s * 1e9,
            "n_semilog": n_s,
            "Is_NoLineal (nA)": popt_sub[0] * 1e9,
            "u(Is_NL) (nA)": err_sub[0] * 1e9,
            "n_NoLineal": popt_sub[1],
            "u(n_NL)": err_sub[1],
            "RMS Residuos (uA)": rms_res * 1e6
        })

df_sens = pd.DataFrame(resultados_sensibilidad)

# =============================================================================
# 5. DIAGNÓSTICO DEL MODELO EXTENDIDO CON RESISTENCIA SERIE Rs (Ecuación 6)
# =============================================================================
def modelo_shockley_Rs(VD, Is, n, Rs):
    """
    Resuelve implícitamente la Ecuación (6): ID = Is * [exp((VD - ID*Rs)/(n*VT)) - 1]
    mediante iteraciones vectorizadas de Newton-Raphson.
    """
    # Semilla segura acotando el exponente para evitar desbordamiento numérico
    ID = Is * (np.exp(np.clip(VD / (n * VT), -50, 40)) - 1.0)
    ID = np.clip(ID, 0.0, 0.05)
    for _ in range(10):
        exp_term = np.exp(np.clip((VD - ID * Rs) / (n * VT), -50, 40))
        f_val = Is * (exp_term - 1.0) - ID
        df_dID = -Is * (Rs / (n * VT)) * exp_term - 1.0
        ID = ID - f_val / df_dID
    return ID

# Probamos ajustar el modelo extendido incluyendo las corrientes más altas (VD >= 0.42 V)
df_alta = df_pos[df_pos["VD"] >= 0.42].copy()
try:
    popt_rs, pcov_rs = curve_fit(
        modelo_shockley_Rs,
        df_alta["VD"],
        df_alta["ID"],
        p0=[Is_nl, n_nl, 2.0],
        bounds=([1e-15, 0.5, 0.0], [1e-3, 5.0, 500.0]),
        maxfev=10000
    )
    Is_rs, n_rs, Rs_est = popt_rs
    u_Is_rs, u_n_rs, u_Rs_est = np.sqrt(np.diag(pcov_rs))
    df_alta["residuos_rs"] = df_alta["ID"] - modelo_shockley_Rs(df_alta["VD"], *popt_rs)
    rs_convergio = True
except Exception as e:
    rs_convergio = False
    print(f"Aviso en ajuste con Rs: {e}")

# =============================================================================
# 6. IMPRESIÓN DE RESULTADOS PARA EL CUADRO 2 Y EL INFORME
# =============================================================================
print("=====================================================================")
print(f"RESULTADOS DEL AJUSTE EN INTERVALO PRINCIPAL [{VD_min:.2f} V, {VD_max:.2f} V]")
print("=====================================================================")
print("1) Método Linealizado Semilogarítmico (Ecuación 4 y 5 - Diagnóstico/Semilla):")
print(f"   Pendiente m = ({m_semilog:.3f} +/- {u_m:.3f}) 1/V")
print(f"   Ordenada  b = ({b_semilog:.3f} +/- {u_b:.3f})")
print(f"   Is inicial  = ({Is_semilog*1e9:.4f} +/- {u_Is_semilog*1e9:.4f}) nA")
print(f"   n inicial   = ({n_semilog:.4f} +/- {u_n_semilog:.4f})\n")

print("2) Ajuste No Lineal Directo de Shockley (Ecuación 1 - Para Cuadro 2):")
print(f"   Is efectiva = ({Is_nl*1e9:.4f} +/- {u_Is_nl*1e9:.4f}) nA  [{Is_nl:.4e} A]")
print(f"   n efectivo  = ({n_nl:.4f} +/- {u_n_nl:.4f})")
print(f"   Desvío estándar de residuos = {df_fit['residuos_nl'].std()*1e6:.2f} uA\n")

print("3) Sensibilidad al cambio del intervalo de ajuste (Punto 6.2.5):")
print(df_sens.to_string(index=False))
print("\n4) Diagnóstico del Modelo Extendido con Resistencia Serie Rs (Ecuación 6):")
if rs_convergio:
    print(f"   Is (modelo Rs) = ({Is_rs*1e9:.4f} +/- {u_Is_rs*1e9:.4f}) nA")
    print(f"   n  (modelo Rs) = ({n_rs:.4f} +/- {u_n_rs:.4f})")
    print(f"   Rs estimada    = ({Rs_est:.2f} +/- {u_Rs_est:.2f}) Ohm")
    print(f"   RMS residuos sin Rs (en VD>=0.42V): {np.sqrt(np.mean(df_eval[df_eval['VD']>=0.42]['residuos_nl']**2))*1e6:.2f} uA")
    print(f"   RMS residuos con Rs (en VD>=0.42V): {np.sqrt(np.mean(df_alta['residuos_rs']**2))*1e6:.2f} uA")

# =============================================================================
# 7. GRÁFICAS DE DATOS, MODELOS Y RESIDUOS (FIGURA DE 4 PANELES)
# =============================================================================
vd_smooth = np.linspace(0.35, df_pos["VD"].max(), 400)
id_smooth_nl = modelo_shockley(vd_smooth, Is_nl, n_nl)
id_smooth_semilog = modelo_shockley(vd_smooth, Is_semilog, n_semilog)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): Escala Lineal (Datos vs Ajuste No Lineal vs Ajuste Semilog)
ax = axs[0, 0]
ax.plot(df_pos["VD"], df_pos["ID"] * 1e3, ".", color="lightgray", markersize=3, label="Datos fuera del intervalo")
ax.plot(df_fit["VD"], df_fit["ID"] * 1e3, ".", color="tab:blue", markersize=3, alpha=0.5, label=f"Datos ajustados [{VD_min}, {VD_max}] V")
ax.plot(vd_smooth, id_smooth_nl * 1e3, "-", color="tab:red", linewidth=2, label=f"Ajuste No Lineal ($n={n_nl:.2f}, I_s={Is_nl*1e9:.2f}$ nA)")
ax.plot(vd_smooth, id_smooth_semilog * 1e3, "--", color="tab:green", linewidth=1.8, label=f"Desde Semilog ($n={n_semilog:.2f}, I_s={Is_semilog*1e9:.2f}$ nA)")
ax.axvline(VD_min, color="k", linestyle=":", alpha=0.6)
ax.axvline(VD_max, color="k", linestyle=":", alpha=0.6)
ax.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax.set_ylabel("Corriente $I_D$ (mA)")
ax.set_title("(a) Ajuste de Shockley en Escala Lineal")
ax.set_ylim(-0.1, df_pos["ID"].max() * 1e3 * 1.05)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

# Panel (b): Escala Semilogarítmica (Recta de Ecuación 4 vs Curva No Lineal)
ax = axs[0, 1]
ax.plot(df_pos["VD"], df_pos["ln_ID"], ".", color="lightgray", markersize=3, label="Datos excluidos")
ax.plot(df_fit["VD"], df_fit["ln_ID"], ".", color="tab:blue", markersize=3, alpha=0.5, label="Datos en intervalo de ajuste")
ax.plot(vd_smooth, m_semilog * vd_smooth + b_semilog, "--", color="tab:green", linewidth=2, label="Regresión lineal de $\\ln(I_D)$ (Ec. 4)")
ax.plot(vd_smooth, np.log(id_smooth_nl), "-", color="tab:red", linewidth=2, label="Ajuste No Lineal visto en semilog")
ax.axvline(VD_min, color="k", linestyle=":", alpha=0.6)
ax.axvline(VD_max, color="k", linestyle=":", alpha=0.6)
ax.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax.set_ylabel("$\\ln(I_D / \\mathrm{1\\,A})$")
ax.set_title("(b) Comparación en Representación Semilogarítmica")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(True, alpha=0.3)

# Panel (c): Residuos dentro del intervalo de ajuste (Ecuación 13)
ax = axs[1, 0]
ax.plot(df_fit["VD"], df_fit["residuos_nl"] * 1e6, ".", color="tab:red", markersize=3, alpha=0.5, label="Residuos Ajuste No Lineal")
ax.plot(df_fit["VD"], df_fit["residuos_semilog"] * 1e6, ".", color="tab:green", markersize=3, alpha=0.4, label="Residuos Modelo Semilog")
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax.set_ylabel("Residuo $r_i = I_{D,i} - I_{modelo}(V_{D,i})$ ($\\mu$A)")
ax.set_title("(c) Residuos en el Intervalo de Ajuste")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

# Panel (d): Residuos en todo el rango de conducción y efecto de Rs a alta corriente
ax = axs[1, 1]
ax.plot(df_eval["VD"], df_eval["residuos_nl"] * 1e6, ".", color="tab:purple", markersize=3, alpha=0.5, label="Shockley simple (extrapolado a todo el rango)")
if rs_convergio:
    ax.plot(df_alta["VD"], df_alta["residuos_rs"] * 1e6, ".", color="tab:orange", markersize=3, alpha=0.5, label=f"Modelo extendido ($R_s = {Rs_est:.1f}\\,\\Omega$)")
ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.axvline(VD_min, color="k", linestyle=":", alpha=0.6, label="Límites intervalo simple")
ax.axvline(VD_max, color="k", linestyle=":", alpha=0.6)
ax.set_xlabel("Tensión en el diodo $V_D$ (V)")
ax.set_ylabel("Residuo $r_i$ ($\\mu$A)")
ax.set_title("(d) Diagnóstico de Validez y Resistencia Serie $R_s$")
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("seccion_6_2_ajuste_shockley.png", dpi=300)
plt.show()