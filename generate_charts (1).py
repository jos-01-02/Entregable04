"""
generate_charts.py
==================
Genera los 4 gráficos de análisis sobre el dataset CICIDS2017 y los guarda
en el directorio de salida cicids2017_outputs/.

Gráficos producidos
-------------------
1. chart1_port_attack_heatmap.png
   Mapa de calor de correlación y contingencia cruzada entre Puertos de
   Destino (Top-20) y perfiles taxonómicos de ataque.

2. chart2_class_distribution.png
   Gráfico de barras — distribución macroscópica de frecuencias de clases
   de tráfico en el dataset CICIDS2017.

3. chart3_confusion_matrix.html  +  chart3_confusion_matrix.png
   Heatmap interactivo (Plotly) de la Matriz de Confusión Normalizada
   generada con el modelo Random Forest del pipeline principal.

4. chart4_metrics_per_class.png
   Gráfico de barras agrupadas de Precisión, Recall y F1-Score por clase
   taxonómica del modelo.

Dependencias
------------
    pip install pandas numpy matplotlib seaborn plotly scikit-learn joblib kaleido
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Plotly para el gráfico interactivo (Chart 3)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("[Aviso] plotly no está instalado. Chart 3 solo se generará en PNG estático.")

# Scikit-learn + joblib para reproducir predicciones (Chart 3)
try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[Aviso] scikit-learn/joblib no disponibles. Chart 3 usará modo estimado.")

warnings.filterwarnings('ignore')

# ─── Rutas ────────────────────────────────────────────────────────────────────
DIRECTORIO_LOCAL  = "./cicids2017_local_data/"
DIRECTORIO_SALIDA = "./cicids2017_outputs/"

ARCHIVOS_CSV = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",  # variante de nombre
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Columnas a excluir del dataset (igual que el pipeline principal)
COLUMNAS_EXCLUIDAS = ['Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'Source Port']
# Nota: 'Destination Port' se excluye del modelo pero la mantenemos aquí para Chart 1.

# Paleta de colores institucional
PALETTE_MAIN  = "Blues"
PALETTE_BARS  = ["#2E86AB", "#E84855", "#3BB273"]   # azul, rojo, verde
sns.set_theme(style="whitegrid", font_scale=1.0)

os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

def _cargar_columnas(columnas_deseadas: list) -> pd.DataFrame:
    """
    Carga solo las columnas indicadas de todos los CSVs disponibles.
    Muy rápido comparado con cargar el dataset completo.
    """
    frames = []
    for nombre in ARCHIVOS_CSV:
        ruta = os.path.join(DIRECTORIO_LOCAL, nombre)
        if not os.path.exists(ruta):
            continue
        try:
            df = pd.read_csv(ruta, encoding='cp1252', low_memory=False,
                             usecols=lambda c: c.strip() in columnas_deseadas)
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            print(f"  [Aviso] No se pudo leer {nombre}: {e}")
    if not frames:
        raise FileNotFoundError(
            f"No se encontró ningún CSV en {DIRECTORIO_LOCAL}. "
            "Asegúrate de ejecutar primero cicids2017_pipeline_local.py."
        )
    return pd.concat(frames, ignore_index=True)


def _cargar_dataset_completo() -> pd.DataFrame:
    """
    Carga el dataset completo (todas las columnas) para reproducir predicciones.
    """
    frames = []
    seen = set()
    for nombre in ARCHIVOS_CSV:
        ruta = os.path.join(DIRECTORIO_LOCAL, nombre)
        # Evitar cargar variantes duplicadas del mismo día
        if not os.path.exists(ruta) or ruta in seen:
            continue
        seen.add(ruta)
        try:
            df = pd.read_csv(ruta, encoding='cp1252', low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
            print(f"  Cargado: {nombre} ({df.shape[0]:,} filas)")
        except Exception as e:
            print(f"  [Aviso] Error leyendo {nombre}: {e}")
    return pd.concat(frames, ignore_index=True)


def _limpiar_df(df: pd.DataFrame, target_col: str = 'Label',
                drop_dest_port: bool = True) -> pd.DataFrame:
    """Replica la limpieza del pipeline principal."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    excluir = COLUMNAS_EXCLUIDAS + (['Destination Port'] if drop_dest_port else [])
    cols_utiles = [c for c in df.columns if c not in excluir]
    return df[cols_utiles]


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 1 — Heatmap de contingencia: Puertos de Destino × Clases de Ataque
# ══════════════════════════════════════════════════════════════════════════════

def chart1_port_attack_heatmap():
    """
    Carga solo 'Destination Port' y 'Label' para construir una tabla de
    contingencia log-normalizada entre los 20 puertos más frecuentes y las
    clases de tráfico del dataset.
    """
    print("\n[Chart 1] Cargando columnas 'Destination Port' y 'Label'...")
    df = _cargar_columnas(['Destination Port', 'Label'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['Label'] = df['Label'].astype(str).str.strip()

    # Top-20 puertos de destino con mayor volumen de tráfico total
    top_puertos = (
        df['Destination Port']
        .value_counts()
        .head(20)
        .index
        .tolist()
    )
    df_top = df[df['Destination Port'].isin(top_puertos)].copy()
    df_top['Destination Port'] = df_top['Destination Port'].astype(int)

    # Tabla de contingencia: filas = clases, columnas = puertos
    tabla = pd.crosstab(df_top['Label'], df_top['Destination Port'])

    # Escala logarítmica para compensar el desequilibrio extremo de clases
    tabla_log = np.log1p(tabla)

    fig, ax = plt.subplots(figsize=(18, 9))
    sns.heatmap(
        tabla_log,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor='white',
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        ax=ax,
        cbar_kws={"label": "log(1 + frecuencia)", "shrink": 0.75}
    )
    ax.set_title(
        "Mapa de Calor de Contingencia Cruzada\n"
        "Puertos de Destino (Top-20) × Perfiles Taxonómicos de Ataque — CICIDS2017",
        fontsize=13, fontweight='bold', pad=14
    )
    ax.set_xlabel("Puerto de Destino", fontsize=11)
    ax.set_ylabel("Clase de Tráfico / Ataque", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()

    ruta = os.path.join(DIRECTORIO_SALIDA, "chart1_port_attack_heatmap.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 2 — Distribución macroscópica de frecuencias de clases de tráfico
# ══════════════════════════════════════════════════════════════════════════════

def chart2_class_distribution():
    """
    Gráfico de barras horizontales mostrando la distribución de instancias
    por clase en el dataset completo CICIDS2017 (escala log en eje X).
    """
    print("\n[Chart 2] Cargando columna 'Label' para distribución de clases...")
    df = _cargar_columnas(['Label'])
    df['Label'] = df['Label'].astype(str).str.strip()

    conteo = df['Label'].value_counts().sort_values(ascending=True)
    total  = conteo.sum()
    porcs  = (conteo / total * 100).round(2)

    # Colores: BENIGN en azul oscuro, resto en escala degradada
    n = len(conteo)
    colors = plt.cm.RdYlBu(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(13, max(7, n * 0.55)))
    bars = ax.barh(conteo.index, conteo.values, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    # Anotación de porcentaje sobre cada barra
    for bar, porc, cnt in zip(bars, porcs.values, conteo.values):
        ax.text(
            bar.get_width() * 1.03,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({porc}%)",
            va='center', ha='left', fontsize=8.5, color='#333333'
        )

    ax.set_xscale('log')
    ax.set_xlabel("Número de instancias (escala logarítmica)", fontsize=11)
    ax.set_ylabel("Clase de Tráfico / Ataque", fontsize=11)
    ax.set_title(
        "Distribución Macroscópica de Frecuencias de Clases de Tráfico\n"
        "Dataset CICIDS2017 — Todos los días de captura",
        fontsize=13, fontweight='bold', pad=14
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    ruta = os.path.join(DIRECTORIO_SALIDA, "chart2_class_distribution.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 3 — Heatmap interactivo de la Matriz de Confusión Normalizada
# ══════════════════════════════════════════════════════════════════════════════

def _obtener_confusion_matrix_desde_modelo():
    """
    Carga el modelo y encoder serializados, reproduce el mismo split 80/20
    con random_state=42 y devuelve (cm_norm, clases).
    """
    ruta_modelo  = os.path.join(DIRECTORIO_SALIDA, "random_forest_cicids2017.pkl")
    ruta_encoder = os.path.join(DIRECTORIO_SALIDA, "label_encoder_cicids2017.pkl")

    if not (os.path.exists(ruta_modelo) and os.path.exists(ruta_encoder)):
        return None, None

    print("  Cargando modelo y encoder guardados...")
    clf     = joblib.load(ruta_modelo)
    encoder = joblib.load(ruta_encoder)

    print("  Cargando dataset completo para reproducir el split de test...")
    df = _cargar_dataset_completo()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    excluir = COLUMNAS_EXCLUIDAS + ['Destination Port']
    cols_utiles = [c for c in df.columns if c not in excluir]
    df = df[cols_utiles]

    df['Label'] = encoder.transform(df['Label'].astype(str).str.strip())
    X = df.drop(columns=['Label'])
    y = df['Label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("  Generando predicciones sobre el conjunto de test...")
    y_pred = clf.predict(X_test)
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    return cm_norm, encoder.classes_


def _estimar_confusion_matrix_desde_reporte():
    """
    Cuando el modelo no está disponible, construye una matriz de confusión
    diagonal aproximada a partir del classification_report.json (recall ≈ TPR).
    """
    ruta_json = os.path.join(DIRECTORIO_SALIDA, "classification_report.json")
    if not os.path.exists(ruta_json):
        return None, None

    with open(ruta_json, encoding='utf-8') as f:
        reporte = json.load(f)

    clases = [k for k, v in reporte.items()
              if isinstance(v, dict) and k not in ('accuracy', 'macro avg', 'weighted avg')]
    n = len(clases)
    cm = np.zeros((n, n))

    for i, clase in enumerate(clases):
        recall = reporte[clase].get('recall', 0.0)
        # Diagonal = recall (fracción predicha correctamente)
        # Residuo se distribuye uniformemente entre las demás clases
        cm[i, i] = recall
        resto = max(0.0, 1.0 - recall) / max(n - 1, 1)
        for j in range(n):
            if j != i:
                cm[i, j] = resto

    return cm, clases


def chart3_confusion_matrix():
    """
    Genera un heatmap interactivo (Plotly HTML) y una versión estática PNG
    de la Matriz de Confusión Normalizada del modelo CICIDS2017.
    """
    print("\n[Chart 3] Construyendo Matriz de Confusión Normalizada...")

    cm, clases = None, None

    # Intentar obtener CM real desde el modelo serializado
    if SKLEARN_OK:
        try:
            cm, clases = _obtener_confusion_matrix_desde_modelo()
        except Exception as e:
            print(f"  [Aviso] No se pudo ejecutar el modelo: {e}")

    # Fallback: CM estimada desde el reporte JSON
    if cm is None:
        print("  Usando CM estimada desde classification_report.json...")
        cm, clases = _estimar_confusion_matrix_desde_reporte()

    if cm is None or clases is None:
        print("  [Error] No hay datos suficientes para generar Chart 3.")
        return

    # ── Versión interactiva con Plotly ────────────────────────────────────────
    if PLOTLY_OK:
        # Redondear para la anotación de celdas
        z_text = [[f"{val:.2f}" for val in row] for row in cm]

        fig_plotly = go.Figure(data=go.Heatmap(
            z=cm,
            x=list(clases),
            y=list(clases),
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale="Blues",
            zmin=0, zmax=1,
            colorbar=dict(title="Proporción", titleside="right"),
        ))
        fig_plotly.update_layout(
            title=dict(
                text="Matriz de Confusión Normalizada — CICIDS2017<br>"
                     "<sup>Random Forest (n_estimators=100, max_depth=10)</sup>",
                x=0.5, xanchor='center', font=dict(size=16)
            ),
            xaxis=dict(title="Predicción", tickangle=-40, tickfont=dict(size=10)),
            yaxis=dict(title="Valor Real", autorange="reversed",
                       tickfont=dict(size=10)),
            width=900, height=820,
            margin=dict(l=130, r=60, t=100, b=140),
            font=dict(family="Arial, sans-serif")
        )

        ruta_html = os.path.join(DIRECTORIO_SALIDA, "chart3_confusion_matrix.html")
        fig_plotly.write_html(ruta_html, full_html=True, include_plotlyjs='cdn')
        print(f"  → Guardado (interactivo): {ruta_html}")

        # Versión PNG mediante kaleido (opcional)
        try:
            ruta_png_plotly = os.path.join(DIRECTORIO_SALIDA, "chart3_confusion_matrix_plotly.png")
            fig_plotly.write_image(ruta_png_plotly, scale=2)
            print(f"  → Guardado (PNG Plotly): {ruta_png_plotly}")
        except Exception:
            pass  # kaleido no instalado — el HTML sigue disponible

    # ── Versión estática con Matplotlib/Seaborn ────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(clases) * 0.75),
                                    max(8, len(clases) * 0.65)))
    mask_annot = cm >= 0.005   # Solo anotar celdas con valor relevante

    # Heatmap base
    sns.heatmap(
        cm, annot=mask_annot, fmt=".2f", cmap="Blues",
        xticklabels=clases, yticklabels=clases,
        vmin=0, vmax=1, linewidths=0.3, linecolor='#cccccc',
        ax=ax, cbar_kws={"label": "Proporción (normalizada por fila)", "shrink": 0.75},
        annot_kws={"size": 7}
    )
    ax.set_title(
        "Matriz de Confusión Normalizada — CICIDS2017\n"
        "Random Forest (n_estimators=100, max_depth=10)",
        fontsize=13, fontweight='bold', pad=14
    )
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Valor Real", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    ruta_png = os.path.join(DIRECTORIO_SALIDA, "chart3_confusion_matrix.png")
    fig.savefig(ruta_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado (PNG estático): {ruta_png}")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO 4 — Barras agrupadas: Precisión, Recall y F1-Score por clase
# ══════════════════════════════════════════════════════════════════════════════

def chart4_metrics_per_class():
    """
    Genera un gráfico de barras agrupadas de alta resolución mostrando
    Precision, Recall y F1-Score para cada clase taxonómica del modelo,
    usando el classification_report.json guardado por el pipeline principal.
    """
    ruta_json = os.path.join(DIRECTORIO_SALIDA, "classification_report.json")
    if not os.path.exists(ruta_json):
        print("\n[Chart 4] No se encontró classification_report.json. Saltando.")
        return

    print("\n[Chart 4] Leyendo classification_report.json...")
    with open(ruta_json, encoding='utf-8') as f:
        reporte = json.load(f)

    # Solo clases reales (excluir promedios de resumen)
    excluir_keys = {'accuracy', 'macro avg', 'weighted avg'}
    clases    = [k for k, v in reporte.items()
                 if isinstance(v, dict) and k not in excluir_keys]
    precision = [reporte[c]['precision'] for c in clases]
    recall    = [reporte[c]['recall']    for c in clases]
    f1        = [reporte[c]['f1-score']  for c in clases]
    support   = [reporte[c]['support']   for c in clases]

    # Limpiar nombres de clases (eliminar caracteres mojibake de encoding)
    def limpiar_nombre(nombre):
        return (nombre
                .replace('ï¿½', '–')
                .replace('\ufffd', '–'))

    clases_limpias = [limpiar_nombre(c) for c in clases]

    # ── Layout ──────────────────────────────────────────────────────────────
    x     = np.arange(len(clases_limpias))
    ancho = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(max(15, len(clases) * 1.15), 11),
                             gridspec_kw={'height_ratios': [3.5, 1]})

    ax = axes[0]
    b1 = ax.bar(x - ancho, precision, ancho, label='Precision',
                color=PALETTE_BARS[0], edgecolor='white', linewidth=0.4)
    b2 = ax.bar(x,          recall,    ancho, label='Recall',
                color=PALETTE_BARS[1], edgecolor='white', linewidth=0.4)
    b3 = ax.bar(x + ancho,  f1,        ancho, label='F1-Score',
                color=PALETTE_BARS[2], edgecolor='white', linewidth=0.4)

    # Anotaciones sobre las barras más bajas (< 0.5) para resaltar problemas
    for bar, val in zip(list(b1) + list(b2) + list(b3),
                        precision + recall + f1):
        if val < 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=7.5,
                color='#cc0000', fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(clases_limpias, rotation=42, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Precisión, Recall y F1-Score por Clase Taxonómica del Modelo\n"
        "Random Forest — Dataset CICIDS2017",
        fontsize=13, fontweight='bold', pad=14
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6,
               label='Umbral 0.90')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Sub-panel: Support (instancias por clase en escala log)
    ax2 = axes[1]
    ax2.bar(x, support, color='#607D8B', alpha=0.7, edgecolor='white', linewidth=0.4)
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clases_limpias, rotation=42, ha='right', fontsize=8)
    ax2.set_ylabel("Support\n(log)", fontsize=9)
    ax2.set_title("Número de instancias en Test (Support)", fontsize=10, pad=6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.grid(axis='y', linestyle='--', alpha=0.35)

    plt.tight_layout(h_pad=3)
    ruta = os.path.join(DIRECTORIO_SALIDA, "chart4_metrics_per_class.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  GENERADOR DE GRÁFICOS — CICIDS2017 ANALYSIS SUITE")
    print("=" * 70)

    try:
        chart1_port_attack_heatmap()
    except Exception as e:
        print(f"  [ERROR Chart 1] {e}")

    try:
        chart2_class_distribution()
    except Exception as e:
        print(f"  [ERROR Chart 2] {e}")

    try:
        chart3_confusion_matrix()
    except Exception as e:
        print(f"  [ERROR Chart 3] {e}")

    try:
        chart4_metrics_per_class()
    except Exception as e:
        print(f"  [ERROR Chart 4] {e}")

    print("\n" + "=" * 70)
    print("  Todos los gráficos guardados en:", os.path.abspath(DIRECTORIO_SALIDA))
    print("=" * 70)
