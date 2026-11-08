import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS MUESTRALES (Ecuaciones 14 y 15)
# =============================================================================
ensayos = [
    {"id": "S1", "A_conf": 1.0, "archivo": "parteB_S1_1V.csv"},
    {"id": "S2", "A_conf": 2.0, "archivo": "parteB_S2_2V.csv"},
    {"id": "S3", "A_conf": 4.0, "archivo": "parteB_S3_4V.csv"},
]

fs_nominal = 10000.0   # Frecuencia de muestreo: 10 kHz
Tb = 1.0               # Duración de cada bloque: 1 s
N_bloque = int(fs_nominal * Tb)  # N = 10000 puntos por bloque (Ec. 14)
n_bloques = 5          # 5 bloques de 1 s sin superposición (total 5 s)
f0_nominal = 100.0     # Frecuencia fundamental nominal: 100 Hz
K_max = 10             # Orden máximo de armónicos para THD_10

# =============================================================================
# 2. FUNCIONES DE PROCESAMIENTO ESPECTRAL (FFT UNILATERAL Y ARMÓNICOS)
# =============================================================================
def cargar_csv_pasco(ruta):
    """Carga el archivo CSV detectando coma o punto decimal automáticamente."""
    df_raw = pd.read_csv(ruta, sep=None, engine="python", decimal=",")
    if df_raw.shape[1] < 3 or df_raw.dtypes.iloc[0] == "object":
        df_raw = pd.read_csv(ruta, sep=None, engine="python", decimal=".")
    df = df_raw.iloc[:, :3].copy()
    df.columns = ["t", "Vs", "VD"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    # Señal de salida analizada espectralmente: VR(t) = Vs(t) - VD(t)
    df["VR"] = df["Vs"] - df["VD"]
    return df


def calcular_fft_unilateral(bloque_senal, fs, usar_hann=True):
    """
    Sustrae el valor medio (DC), aplica ventana (Hann o Rectangular),
    corrige por ganancia coherente y devuelve el espectro unilateral normalizado.
    """
    N = len(bloque_senal)
    
    # Paso 6.3.2: Calcular y conservar componente continua (DC) y sustraerla
    dc_val = np.mean(bloque_senal)
    senal_sin_dc = bloque_senal - dc_val

    # Paso 6.3.3: Aplicación de ventana y corrección por ganancia coherente
    if usar_hann:
        # Ventana de Hann periódica para análisis espectral FFT
        n_idx = np.arange(N)
        ventana = 0.5 * (1.0 - np.cos(2.0 * np.pi * n_idx / N))
        ganancia_coherente = np.mean(ventana)  # Exactamente 0.5
    else:
        ventana = np.ones(N)
        ganancia_coherente = 1.0

    senal_ventaneada = senal_sin_dc * ventana

    # Paso 6.3.4: Cálculo de FFT unilateral y normalización de amplitud
    fft_compleja = np.fft.rfft(senal_ventaneada)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Normalización base dividiendo por N y por la ganancia coherente de la ventana
    amplitudes = np.abs(fft_compleja) / (N * ganancia_coherente)

    # Duplicamos las frecuencias positivas interiores para el espectro unilateral,
    # evitando duplicar la componente continua (índice 0) y Nyquist (índice -1 si N es par)
    if N % 2 == 0:
        amplitudes[1:-1] *= 2.0
    else:
        amplitudes[1:] *= 2.0

    return freqs, amplitudes, dc_val


def extraer_armonicos_y_thd(freqs, amplitudes, f0_busqueda=100.0, K=10, tol_f0=3.0, tol_arm=2.0):
    """
    Paso 6.3.5 y 6.3.6: Localiza la fundamental cerca de f0 y busca los armónicos
    estrictamente en múltiplos k * f0_medida. Calcula A_1...A_K y THD_K (Ec. 8).
    """
    # 1. Buscar la frecuencia fundamental real medida cerca de f0_busqueda (100 Hz)
    mask_f0 = (freqs >= f0_busqueda - tol_f0) & (freqs <= f0_busqueda + tol_f0)
    idx_f0_local = np.argmax(amplitudes[mask_f0])
    freqs_f0 = freqs[mask_f0]
    f0_medida = freqs_f0[idx_f0_local]

    # 2. Extraer amplitudes A_1 hasta A_K buscando en el entorno de k * f0_medida
    A_k = np.zeros(K)
    freqs_k = np.zeros(K)
    for k in range(1, K + 1):
        fk_esperada = k * f0_medida
        mask_k = (freqs >= fk_esperada - tol_arm) & (freqs <= fk_esperada + tol_arm)
        A_k[k - 1] = np.max(amplitudes[mask_k])
        freqs_k[k - 1] = freqs[mask_k][np.argmax(amplitudes[mask_k])]

    # 3. Estimar piso de ruido (mediana del espectro fuera de los armónicos)
    piso_ruido = np.median(amplitudes[(freqs > 50) & (freqs < 1500)])

    # 4. Cálculo de THD_K (Ecuación 8): sqrt(A_2^2 + ... + A_K^2) / A_1
    A1 = A_k[0]
    if A1 > 5.0 * piso_ruido:
        thd_k = np.sqrt(np.sum(A_k[1:] ** 2)) / A1
    else:
        thd_k = np.nan  # No estable si A1 no se distingue del ruido

    return f0_medida, A_k, freqs_k, thd_k, piso_ruido


def resumen_mediana_rango(valores):
    """Paso 6.3.7: Calcula la mediana, el intervalo [mín, máx] y el IQR de los 5 bloques."""
    med = np.nanmedian(valores)
    vmin, vmax = np.nanmin(valores), np.nanmax(valores)
    q25, q75 = np.nanpercentile(valores, [25, 75])
    iqr = q75 - q25
    return med, vmin, vmax, iqr

# =============================================================================
# 3. PROCESAMIENTO POR BLOQUES DE LOS 3 ENSAYOS (S1, S2, S3)
# =============================================================================
tabla_cuadro1 = []
tabla_cuadro3 = []
tabla_entrada_vs = []
espectros_guardados = {}

for item in ensayos:
    eid = item["id"]
    A_conf = item["A_conf"]
    df = cargar_csv_pasco(item["archivo"])

    # Verificación de frecuencia de muestreo real
    dt = np.median(np.diff(df["t"]))
    fs_real = 1.0 / dt

    # Listas para almacenar resultados de los 5 bloques de 1 s
    A_in_bloques, f0_bloques, thd_in_bloques = [], [], []
    dc_out_bloques, A1_out_bloques, A2_A1_bloques, A3_A1_bloques, thd_out_bloques = [], [], [], [], []
    Ak_out_matriz = []

    for b in range(n_bloques):
        inicio = b * N_bloque
        fin = (b + 1) * N_bloque
        bloque_Vs = df["Vs"].iloc[inicio:fin].to_numpy()
        bloque_VR = df["VR"].iloc[inicio:fin].to_numpy()

        # --- Análisis de la ENTRADA Vs(t) (para separar distorsión del generador) ---
        freqs_in, amps_in, dc_in = calcular_fft_unilateral(bloque_Vs, fs_real, usar_hann=True)
        f0_in, Ak_in, _, thd_in, _ = extraer_armonicos_y_thd(freqs_in, amps_in, f0_nominal, K_max)

        # --- Análisis de la SALIDA VR(t) ---
        freqs_out, amps_out, dc_out = calcular_fft_unilateral(bloque_VR, fs_real, usar_hann=True)
        # Usamos la f0_in medida en el generador como referencia limpia de frecuencia fundamental
        _, Ak_out, fk_out, thd_out, ruido_out = extraer_armonicos_y_thd(
            freqs_out, amps_out, f0_busqueda=f0_in, K=K_max
        )

        # Guardamos métricas del bloque
        A_in_bloques.append(Ak_in[0])  # Amplitud fundamental real medida en la entrada
        f0_bloques.append(f0_in)
        thd_in_bloques.append(thd_in)

        dc_out_bloques.append(dc_out)
        A1_out_bloques.append(Ak_out[0])
        A2_A1_bloques.append(Ak_out[1] / Ak_out[0])
        A3_A1_bloques.append(Ak_out[2] / Ak_out[0])
        thd_out_bloques.append(thd_out)
        Ak_out_matriz.append(Ak_out)

        # Guardamos el Bloque 1 (b == 0) para las figuras de diagnóstico y comparación
        if b == 0:
            _, amps_out_rect, _ = calcular_fft_unilateral(bloque_VR, fs_real, usar_hann=False)
            espectros_guardados[eid] = {
                "freqs": freqs_out,
                "amps_in_hann": amps_in,
                "amps_out_hann": amps_out,
                "amps_out_rect": amps_out_rect,
                "dc_out": dc_out,
                "Ak_out": Ak_out,
                "fk_out": fk_out,
                "thd_out": thd_out,
                "A_med": Ak_in[0]
            }

    # Estadísticas sobre los 5 bloques (Mediana y Dispersión Mín-Máx / IQR)
    A_med, A_min, A_max, A_iqr = resumen_mediana_rango(A_in_bloques)
    f0_med, _, _, _ = resumen_mediana_rango(f0_bloques)
    thdin_med, thdin_min, thdin_max, _ = resumen_mediana_rango(thd_in_bloques)

    dc_med, dc_min, dc_max, dc_iqr = resumen_mediana_rango(dc_out_bloques)
    A1_med, A1_min, A1_max, A1_iqr = resumen_mediana_rango(A1_out_bloques)
    r21_med, r21_min, r21_max, _ = resumen_mediana_rango(A2_A1_bloques)
    r31_med, r31_min, r31_max, _ = resumen_mediana_rango(A3_A1_bloques)
    thd_med, thd_min, thd_max, thd_iqr = resumen_mediana_rango(thd_out_bloques)

    tabla_cuadro1.append({
        "Ensayo": eid,
        "A config (V)": A_conf,
        "A medida (V) [Mediana]": f"{A_med:.4f} [{A_min:.4f}, {A_max:.4f}]",
        "f0 medida (Hz)": f"{f0_med:.2f}",
        "fs real (Hz)": f"{fs_real:.1f}",
        "THD_10 Entrada (%)": f"{thdin_med*100:.3f}%"
    })

    tabla_cuadro3.append({
        "Ensayo": eid,
        "A medida (V)": f"{A_med:.4f}",
        "DC (V)": f"{dc_med:.4f} [{dc_min:.4f}, {dc_max:.4f}]",
        "A1 (V)": f"{A1_med:.4f} [{A1_min:.4f}, {A1_max:.4f}]",
        "A2 / A1": f"{r21_med:.4f}",
        "A3 / A1": f"{r31_med:.4f}",
        "THD_10 (adim)": f"{thd_med:.4f} (min={thd_min:.4f}, max={thd_max:.4f}, IQR={thd_iqr:.4f})",
        "THD_10 (%)": f"{thd_med*100:.2f}%"
    })

# =============================================================================
# 4. IMPRESIÓN DE TABLAS EN CONSOLA (CUADRO 1, CUADRO 3 Y RESOLUCIÓN)
# =============================================================================
print("==========================================================================")
print("PARÁMETROS DE RESOLUCIÓN ESPECTRAL (Ecuaciones 14 y 15)")
print("==========================================================================")
print(f"Frecuencia de muestreo: fs = {fs_nominal:.0f} Hz")
print(f"Duración de bloque:     Tb = {Tb:.1f} s  --> Muestras por bloque: N = {N_bloque}")
print(f"Resolución espectral:   df = fs / N = 1 / Tb = {fs_nominal/N_bloque:.2f} Hz")
print(f"Frecuencia de Nyquist:  f_Nyq = fs / 2 = {fs_nominal/2:.0f} Hz\n")

print("==========================================================================")
print("DATOS PARA COMPLETAR EL CUADRO 1 (Y DISTORSIÓN PROPIA DEL GENERADOR)")
print("==========================================================================")
print(pd.DataFrame(tabla_cuadro1).to_string(index=False))

print("\n==========================================================================")
print("DATOS PARA COMPLETAR EL CUADRO 3 (RESUMEN ESPECTRAL DE LA SALIDA VR)")
print("==========================================================================")
print(pd.DataFrame(tabla_cuadro3).to_string(index=False))

# =============================================================================
# 5. GRÁFICAS DE LA SECCIÓN 6.3: VENTANA DE HANN vs RECTANGULAR Y ARMÓNICOS
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel Izquierdo: Comprobación Punto 6.3.3 (Hann vs Sin Ventana en Ensayo S3 - 4 V)
esp_s3 = espectros_guardados["S3"]
freqs = esp_s3["freqs"]
mask_1200 = (freqs >= 0) & (freqs <= 1150)

# Graficamos en escala logarítmica para ver tanto los picos como la fuga espectral en el piso
ax1.semilogy(freqs[mask_1200], esp_s3["amps_out_rect"][mask_1200], "-", color="tab:orange", alpha=0.7, linewidth=1.2, label="Sin ventana (Rectangular)")
ax1.semilogy(freqs[mask_1200], esp_s3["amps_out_hann"][mask_1200], "-", color="tab:blue", linewidth=1.5, label="Ventana de Hann (corregida por 0.5)")
ax1.plot(esp_s3["fk_out"], esp_s3["Ak_out"], "ro", markersize=5, label="Armónicos $A_1 \\dots A_{10}$ detectados")
ax1.set_xlabel("Frecuencia $f$ (Hz)")
ax1.set_ylabel("Amplitud unilateral de $V_R$ (V) [Escala Log]")
ax1.set_title("Efecto del Ventaneo en la FFT (Ensayo S3, Bloque 1)")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", alpha=0.3)

# Panel Derecho: Zoom alrededor de la fundamental (100 Hz) para ver ancho de pico y fuga lateral
mask_zoom = (freqs >= 75) & (freqs <= 125)
ax2.semilogy(freqs[mask_zoom], esp_s3["amps_out_rect"][mask_zoom], "o-", color="tab:orange", markersize=4, alpha=0.8, label="Sin ventana (Fuga en colas laterales)")
ax2.semilogy(freqs[mask_zoom], esp_s3["amps_out_hann"][mask_zoom], "s-", color="tab:blue", markersize=4, label="Ventana de Hann (Lóbulo más ancho, menor fuga)")
ax2.set_xlabel("Frecuencia $f$ (Hz)")
ax2.set_ylabel("Amplitud unilateral de $V_R$ (V) [Escala Log]")
ax2.set_title("Zoom en Fundamental ($f_0 \\approx 100$ Hz): Fuga vs. Anchura de Pico")
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("seccion_6_3_ventana_y_espectro.png", dpi=300)
plt.show()