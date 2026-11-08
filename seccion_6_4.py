import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARÁMETROS Y LISTA DE ENSAYOS (Parte B)
# =============================================================================
ensayos = [
    {"id": "S1", "A_conf": 1.0, "archivo": "parteB_S1_1V.csv"},
    {"id": "S2", "A_conf": 2.0, "archivo": "parteB_S2_2V.csv"},
    {"id": "S3", "A_conf": 4.0, "archivo": "parteB_S3_4V.csv"},
]

fs_nominal = 10000.0   # Frecuencia de muestreo: 10 kHz
Tb = 1.0               # Duración de cada bloque: 1 s
N_bloque = int(fs_nominal * Tb)  # 10000 muestras por bloque
n_bloques = 5          # 5 bloques por registro de 5 s
f0_nominal = 100.0     # Frecuencia fundamental nominal: 100 Hz
K_max = 10             # Hasta el décimo armónico (1000 Hz)
n_periodos_vista = 5   # Cantidad de períodos a mostrar en el dominio temporal (50 ms)

# =============================================================================
# 2. FUNCIONES AUXILIARES DE CARGA Y FFT UNILATERAL CON VENTANA DE HANN
# =============================================================================
def cargar_csv_pasco(ruta):
    """Carga el CSV de PASCO detectando coma o punto decimal y calcula VR(t)."""
    df_raw = pd.read_csv(ruta, sep=None, engine="python", decimal=",")
    if df_raw.shape[1] < 3 or df_raw.dtypes.iloc[0] == "object":
        df_raw = pd.read_csv(ruta, sep=None, engine="python", decimal=".")
    df = df_raw.iloc[:, :3].copy()
    df.columns = ["t", "Vs", "VD"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    df["VR"] = df["Vs"] - df["VD"]  # Señal de salida proporcional a la corriente
    return df


def fft_hann_unilateral(senal, fs):
    """Sustrae DC, aplica ventana de Hann corregida y devuelve espectro unilateral."""
    N = len(senal)
    dc = np.mean(senal)
    senal_ac = senal - dc

    n_idx = np.arange(N)
    ventana = 0.5 * (1.0 - np.cos(2.0 * np.pi * n_idx / N))
    ganancia_coh = np.mean(ventana)  # 0.5

    espectro_c = np.fft.rfft(senal_ac * ventana)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    amps = np.abs(espectro_c) / (N * ganancia_coh)

    if N % 2 == 0:
        amps[1:-1] *= 2.0
    else:
        amps[1:] *= 2.0

    return freqs, amps, dc


def medir_armonicos(freqs, amps, f0_ref=100.0, K=10, tol_f0=3.0, tol_k=2.0):
    """Extrae f0, las amplitudes A_1...A_K en múltiplos de f0 y calcula THD_K."""
    mask_f0 = (freqs >= f0_ref - tol_f0) & (freqs <= f0_ref + tol_f0)
    f0_med = freqs[mask_f0][np.argmax(amps[mask_f0])]

    Ak = np.zeros(K)
    fk = np.zeros(K)
    for k in range(1, K + 1):
        f_esp = k * f0_med
        mask_k = (freqs >= f_esp - tol_k) & (freqs <= f_esp + tol_k)
        idx_max = np.argmax(amps[mask_k])
        Ak[k - 1] = amps[mask_k][idx_max]
        fk[k - 1] = freqs[mask_k][idx_max]

    piso_ruido = np.median(amps[(freqs > 50) & (freqs < 1500)])
    thd = np.sqrt(np.sum(Ak[1:] ** 2)) / Ak[0] if Ak[0] > 5.0 * piso_ruido else np.nan
    return f0_med, Ak, fk, thd

# =============================================================================
# 3. PROCESAMIENTO DE CADA AMPLITUD Y GENERACIÓN DE LAS 3 FIGURAS INDIVIDUALES
# =============================================================================
resumen_global = []

for item in ensayos:
    eid = item["id"]
    A_conf = item["A_conf"]
    df = cargar_csv_pasco(item["archivo"])

    dt = np.median(np.diff(df["t"]))
    fs_real = 1.0 / dt

    # Listas para acumular los 5 bloques de 1 s
    A_in_b, dc_out_b, A1_out_b, thd_out_b, frac_cond_b = [], [], [], [], []
    espectro_b1 = {}

    # Umbral operativo en VR para estimar la fracción del ciclo en conducción (> 20 mV)
    umbral_VR_cond = 0.02

    for b in range(n_bloques):
        ini = b * N_bloque
        fin = (b + 1) * N_bloque
        v_in = df["Vs"].iloc[ini:fin].to_numpy()
        v_out = df["VR"].iloc[ini:fin].to_numpy()

        # FFT de entrada y de salida
        f_in, amps_in, dc_in = fft_hann_unilateral(v_in, fs_real)
        f0_in, Ak_in, fk_in, thd_in = medir_armonicos(f_in, amps_in, f0_nominal, K_max)

        f_out, amps_out, dc_out = fft_hann_unilateral(v_out, fs_real)
        _, Ak_out, fk_out, thd_out = medir_armonicos(f_out, amps_out, f0_in, K_max)

        # Fracción del ciclo en que el diodo conduce (VR claramente positiva)
        frac_cond = np.mean(v_out > umbral_VR_cond) * 100.0

        A_in_b.append(Ak_in[0])
        dc_out_b.append(dc_out)
        A1_out_b.append(Ak_out[0])
        thd_out_b.append(thd_out)
        frac_cond_b.append(frac_cond)

        if b == 0:
            espectro_b1 = {
                "freqs": f_out,
                "amps_in": amps_in,
                "amps_out": amps_out,
                "fk_in": fk_in,
                "Ak_in": Ak_in,
                "fk_out": fk_out,
                "Ak_out": Ak_out,
                "thd_in": thd_in
            }

    # Estadísticos entre los 5 bloques (Mediana y extremos Mín-Máx)
    A_med = np.median(A_in_b)
    A_min, A_max = np.min(A_in_b), np.max(A_in_b)

    dc_med = np.median(dc_out_b)
    A1_med = np.median(A1_out_b)
    thd_med = np.median(thd_out_b)
    thd_min, thd_max = np.min(thd_out_b), np.max(thd_out_b)
    frac_med = np.median(frac_cond_b)

    resumen_global.append({
        "id": eid,
        "A_conf": A_conf,
        "A_med": A_med,
        "A_err_inf": A_med - A_min,
        "A_err_sup": A_max - A_med,
        "dc_med": dc_med,
        "A1_med": A1_med,
        "thd_med": thd_med,
        "thd_err_inf": thd_med - thd_min,
        "thd_err_sup": thd_max - thd_med,
        "frac_cond_pct": frac_med
    })

    # -------------------------------------------------------------------------
    # FIGURA DE LA SECCIÓN 6.4 PARA CADA AMPLITUD (3 PANELES: TIEMPO + 2 ESPECTROS)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    # Panel Superior (ocupa las 2 columnas): Dominio Temporal (5 períodos = 0.05 s)
    ax_t = fig.add_subplot(gs[0, :])
    t_max_vista = df["t"].iloc[0] + (n_periodos_vista / f0_nominal)
    df_vista = df[df["t"] <= t_max_vista]
    t_ms = (df_vista["t"] - df_vista["t"].iloc[0]) * 1e3

    ax_t.plot(t_ms, df_vista["Vs"], "-", color="tab:blue", linewidth=1.8, label="Entrada $V_s(t)$")
    ax_t.plot(t_ms, df_vista["VR"], "-", color="tab:red", linewidth=2.0, label="Salida $V_R(t) = V_s(t) - V_D(t)$")
    ax_t.axhline(dc_med, color="tab:red", linestyle="--", alpha=0.7, label=f"Nivel DC de salida = {dc_med:.3f} V")
    ax_t.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax_t.set_xlabel("Tiempo $t$ (ms)")
    ax_t.set_ylabel("Tensión (V)")
    ax_t.set_title(
        f"Ensayo {eid} ($A_{{config}} = {A_conf:.1f}$ V, $A_{{medida}} = {A_med:.3f}$ V) — Dominio Temporal ({n_periodos_vista} períodos)"
    )
    ax_t.legend(loc="upper right", ncol=3, fontsize=9.5)
    ax_t.grid(True, alpha=0.3)

    # Filtro de frecuencias hasta 1100 Hz (cubre hasta el 10° armónico en 1000 Hz)
    freqs = espectro_b1["freqs"]
    mask_f = (freqs >= 0) & (freqs <= 1100)

    # Panel Inferior Izquierdo: Espectro de Amplitud de la ENTRADA Vs(t)
    ax_in = fig.add_subplot(gs[1, 0])
    ax_in.plot(freqs[mask_f], espectro_b1["amps_in"][mask_f], "-", color="tab:blue", linewidth=1.2)
    ax_in.plot(espectro_b1["fk_in"], espectro_b1["Ak_in"], "bo", markersize=5, label="Armónicos $k f_0$")
    ax_in.set_xlabel("Frecuencia $f$ (Hz)")
    ax_in.set_ylabel("Amplitud unilateral de $V_s$ (V)")
    ax_in.set_title("Espectro de Amplitud de la Entrada $V_s(f)$")
    ax_in.text(
        0.96, 0.92,
        f"$A_{{1,in}} = {A_med:.3f}$ V\n$\\mathrm{{THD}}_{{10,in}} = {espectro_b1['thd_in']*100:.2f}\\%$",
        transform=ax_in.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="aliceblue", edgecolor="tab:blue", alpha=0.9),
        fontsize=9.5
    )
    ax_in.set_xlim(0, 1100)
    ax_in.grid(True, alpha=0.3)

    # Panel Inferior Derecho: Espectro de Amplitud de la SALIDA VR(t) hasta 10° armónico
    ax_out = fig.add_subplot(gs[1, 1])
    ax_out.plot(freqs[mask_f], espectro_b1["amps_out"][mask_f], "-", color="tab:red", linewidth=1.2)
    ax_out.plot(espectro_b1["fk_out"], espectro_b1["Ak_out"], "ro", markersize=5, label="Armónicos $A_1 \\dots A_{10}$")
    ax_out.set_xlabel("Frecuencia $f$ (Hz)")
    ax_out.set_ylabel("Amplitud unilateral de $V_R$ (V)")
    ax_out.set_title("Espectro de Amplitud de la Salida $V_R(f)$ (hasta $10 f_0$)")

    # Cuadro de texto obligatorio con DC, A1 y THD_10 (Mediana de los 5 bloques)
    texto_metricas = (
        f"Valores de Salida (Mediana 5 bloques):\n"
        f"• Componente DC = {dc_med:.4f} V\n"
        f"• Fundamental $A_1$ = {A1_med:.4f} V\n"
        f"• $\\mathrm{{THD}}_{{10}}$ = {thd_med:.4f} ({thd_med*100:.2f}%)\n"
        f"• Conducción $\\approx$ {frac_med:.1f}% del ciclo"
    )
    ax_out.text(
        0.96, 0.92, texto_metricas,
        transform=ax_out.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="tab:red", alpha=0.9),
        fontsize=9.5
    )
    ax_out.set_xlim(0, 1100)
    ax_out.grid(True, alpha=0.3)

    plt.tight_layout()
    nombre_fig = f"seccion_6_4_comparacion_{eid}.png"
    plt.savefig(nombre_fig, dpi=300)
    plt.show()

# =============================================================================
# 4. FIGURA FINAL: THD_10 vs AMPLITUD PICO MEDIDA Y RELACIÓN CON LA CONDUCCIÓN
# =============================================================================
df_res = pd.DataFrame(resumen_global)

fig, (ax_thd, ax_cond) = plt.subplots(1, 2, figsize=(13, 5))

# Panel Izquierdo: THD_10 frente a la amplitud pico medida (sin imponer tendencia artificial)
yerr_thd = [df_res["thd_err_inf"] * 100, df_res["thd_err_sup"] * 100]
xerr_A = [df_res["A_err_inf"], df_res["A_err_sup"]]

ax_thd.errorbar(
    df_res["A_med"], df_res["thd_med"] * 100,
    xerr=xerr_A, yerr=yerr_thd,
    fmt="o", color="tab:red", markersize=8, capsize=5, linewidth=1.5,
    label="Mediana $\\mathrm{THD}_{10}$ (barras: mín–máx en 5 bloques)"
)
for _, fila in df_res.iterrows():
    ax_thd.annotate(
        f"{fila['id']} ({fila['thd_med']*100:.1f}%)",
        (fila["A_med"], fila["thd_med"] * 100),
        textcoords="offset points", xytext=(8, 6), fontsize=9.5, fontweight="bold"
    )

ax_thd.axhline(43.5, color="gray", linestyle=":", linewidth=1.2, label="Referencia rectificador media onda ideal ($\\approx 43.5\\%$)")
ax_thd.set_xlabel("Amplitud pico medida de entrada $A_{medida}$ (V)")
ax_thd.set_ylabel("$\\mathrm{THD}_{10}$ de la salida $V_R$ (%)")
ax_thd.set_title("Distorsión Armónica Total ($\\mathrm{THD}_{10}$) vs. Amplitud de Entrada")
ax_thd.set_xlim(0.5, df_res["A_med"].max() + 0.5)
ax_thd.legend(fontsize=8.5, loc="upper right")
ax_thd.grid(True, alpha=0.3)

# Panel Derecho: Grado de conducción temporal y Nivel DC vs Amplitud medida
color_c1 = "tab:purple"
color_c2 = "tab:green"
ax_cond.plot(df_res["A_med"], df_res["frac_cond_pct"], "s--", color=color_c1, markersize=7, label="Fracción del ciclo en conducción (%)")
ax_cond.axhline(50.0, color=color_c1, linestyle=":", alpha=0.6, label="Límite media onda ideal (50% del ciclo)")
ax_cond.set_xlabel("Amplitud pico medida de entrada $A_{medida}$ (V)")
ax_cond.set_ylabel("Fracción del período con diodo conduciendo (%)", color=color_c1)
ax_cond.tick_params(axis="y", labelcolor=color_c1)
ax_cond.set_ylim(0, 60)
ax_cond.grid(True, alpha=0.3)

ax_dc = ax_cond.twinx()
ax_dc.plot(df_res["A_med"], df_res["dc_med"], "^-", color=color_c2, markersize=7, label="Componente continua DC (V)")
ax_dc.set_ylabel("Componente continua DC en $V_R$ (V)", color=color_c2)
ax_dc.tick_params(axis="y", labelcolor=color_c2)

ax_cond.set_title("Relación con el Dominio Temporal: Conducción y Rectificación")
# Unificamos leyendas de ambos ejes verticales
lines1, labels1 = ax_cond.get_legend_handles_labels()
lines2, labels2 = ax_dc.get_legend_handles_labels()
ax_cond.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8.5)

plt.tight_layout()
plt.savefig("seccion_6_4_THD_vs_Amplitud.png", dpi=300)
plt.show()