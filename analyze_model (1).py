"""
analyze_model.py
================
Análisis exhaustivo del modelo Random Forest serializado (random_forest_cicids2017.pkl)
para presentación en documento de maestría.

Salidas generadas en cicids2017_outputs/thesis/
───────────────────────────────────────────────
  model_summary.txt                — Ficha técnica completa del modelo
  feature_importance_top20.png     — Top-20 variables más importantes (Gini)
  feature_importance_all.csv       — Tabla completa de importancias (exportable)
  feature_importance_cumulative.png— Curva acumulativa de importancia (Pareto)
  decision_tree_sample.png         — Visualización de un árbol individual del ensemble
  tree_depth_distribution.png      — Distribución de profundidades reales del bosque
  roc_curves.png                   — Curvas ROC multiclase (One-vs-Rest)
  learning_insights.txt            — Interpretación académica en texto

Dependencias: scikit-learn, joblib, matplotlib, seaborn, pandas, numpy
"""

import os
import json
import warnings
import textwrap
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.tree import export_graphviz, plot_tree
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# ─── Rutas ────────────────────────────────────────────────────────────────────
DIRECTORIO_LOCAL  = "./cicids2017_local_data/"
DIRECTORIO_SALIDA = "./cicids2017_outputs/"
DIR_TESIS         = os.path.join(DIRECTORIO_SALIDA, "thesis")
COLUMNAS_EXCLUIDAS = [
    'Timestamp', 'Flow ID', 'Source IP', 'Destination IP',
    'Source Port', 'Destination Port'
]
ARCHIVOS_CSV = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

sns.set_theme(style="whitegrid", font_scale=1.05)
os.makedirs(DIR_TESIS, exist_ok=True)


def _limpiar_nombre(nombre: str) -> str:
    return nombre.replace('ï¿½', '–').replace('\ufffd', '–')


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_artefactos():
    ruta_clf = os.path.join(DIRECTORIO_SALIDA, "random_forest_cicids2017.pkl")
    ruta_enc = os.path.join(DIRECTORIO_SALIDA, "label_encoder_cicids2017.pkl")
    if not os.path.exists(ruta_clf) or not os.path.exists(ruta_enc):
        raise FileNotFoundError(
            "No se encontraron los archivos .pkl en cicids2017_outputs/.\n"
            "Ejecuta primero: py cicids2017_pipeline_local.py"
        )
    clf = joblib.load(ruta_clf)
    enc = joblib.load(ruta_enc)
    print(f"[OK] Modelo cargado — {clf.n_estimators} árboles, max_depth={clf.max_depth}")
    print(f"[OK] Encoder cargado — {len(enc.classes_)} clases")
    return clf, enc


def cargar_feature_names() -> list:
    """
    Extrae los nombres de las 77 características usadas por el modelo,
    leyendo un CSV de muestra y aplicando la misma exclusión del pipeline.
    """
    seen = set()
    for nombre in ARCHIVOS_CSV:
        ruta = os.path.join(DIRECTORIO_LOCAL, nombre)
        if not os.path.exists(ruta) or ruta in seen:
            continue
        seen.add(ruta)
        try:
            df = pd.read_csv(ruta, encoding='cp1252', nrows=1)
            df.columns = df.columns.str.strip()
            cols = [c for c in df.columns if c not in COLUMNAS_EXCLUIDAS + ['Label']]
            print(f"[OK] Nombres de features extraídos desde: {nombre} ({len(cols)} columnas)")
            return cols
        except Exception:
            continue
    raise FileNotFoundError("No se pudo leer ningún CSV para extraer nombres de features.")


def cargar_dataset_test(enc: LabelEncoder, feature_names: list):
    """
    Reproduce el split de test 80/20 estratificado con random_state=42,
    idéntico al usado en el pipeline de entrenamiento.
    """
    frames = []
    seen = set()
    for nombre in ARCHIVOS_CSV:
        ruta = os.path.join(DIRECTORIO_LOCAL, nombre)
        if not os.path.exists(ruta) or ruta in seen:
            continue
        seen.add(ruta)
        try:
            df = pd.read_csv(ruta, encoding='cp1252', low_memory=False)
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            print(f"  [Aviso] {nombre}: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    cols_utiles = feature_names + ['Label']
    df = df[[c for c in cols_utiles if c in df.columns]]
    df['Label'] = enc.transform(df['Label'].astype(str).str.strip())

    X = df[feature_names]
    y = df['Label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[OK] Conjunto de test reproducido — {X_test.shape[0]:,} instancias")
    return X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 1 — Ficha técnica del modelo
# ══════════════════════════════════════════════════════════════════════════════

def generar_model_summary(clf, enc: LabelEncoder, feature_names: list):
    print("\n[1/7] Generando ficha técnica del modelo...")

    params = clf.get_params()
    profundidades  = [t.get_depth()      for t in clf.estimators_]
    nodos          = [t.tree_.node_count for t in clf.estimators_]
    hojas          = [t.tree_.n_node_samples[t.tree_.children_left == -1].shape[0]
                      for t in clf.estimators_]
    clases_limpias = [_limpiar_nombre(c) for c in enc.classes_]

    # Cargar métricas de resumen desde el reporte JSON
    reporte_json = os.path.join(DIRECTORIO_SALIDA, "classification_report.json")
    accuracy = macro_f1 = weighted_f1 = "N/A"
    if os.path.exists(reporte_json):
        with open(reporte_json, encoding='utf-8') as f:
            rpt = json.load(f)
        accuracy    = f"{rpt.get('accuracy', 'N/A'):.6f}" if isinstance(rpt.get('accuracy'), float) else "N/A"
        macro_f1    = f"{rpt.get('macro avg', {}).get('f1-score', 'N/A'):.6f}"
        weighted_f1 = f"{rpt.get('weighted avg', {}).get('f1-score', 'N/A'):.6f}"

    resumen = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║         FICHA TÉCNICA DEL MODELO — MAESTRÍA EN CIBERSEGURIDAD       ║
    ╚══════════════════════════════════════════════════════════════════════╝

    ─── ALGORITMO ────────────────────────────────────────────────────────
    Tipo              : Random Forest Classifier (Ensemble Bagging)
    Implementación    : scikit-learn (sklearn.ensemble.RandomForestClassifier)
    Criterio de split : {params['criterion'].upper()} (Gini Impurity)

    ─── HIPERPARÁMETROS ──────────────────────────────────────────────────
    n_estimators      : {params['n_estimators']}  (número de árboles de decisión)
    max_depth         : {params['max_depth']}   (profundidad máxima por árbol)
    max_features      : {params['max_features']}  (√n_features por split)
    min_samples_split : {params['min_samples_split']}     (mín. muestras para dividir nodo)
    min_samples_leaf  : {params['min_samples_leaf']}     (mín. muestras en nodo hoja)
    bootstrap         : {params['bootstrap']}  (muestreo con reemplazo — Bagging)
    random_state      : {params['random_state']}   (semilla de reproducibilidad)
    n_jobs            : {params['n_jobs']}   (paralelismo total de núcleos CPU)

    ─── DATASET Y PARTICIÓN ─────────────────────────────────────────────
    Dataset           : CICIDS2017 (Canadian Institute for Cybersecurity)
    Días de captura   : 5 (Lunes a Viernes — semana laboral completa)
    Partición         : 80% entrenamiento / 20% prueba (estratificada)
    Características   : {len(feature_names)} variables de flujo de red (NetFlow/CICFlowMeter)
    Clases objetivo   : {len(enc.classes_)} ({', '.join(clases_limpias)})

    ─── ESTRUCTURA DEL BOSQUE ───────────────────────────────────────────
    Profundidad real promedio : {np.mean(profundidades):.2f} ± {np.std(profundidades):.2f}
    Profundidad real máxima   : {max(profundidades)}
    Nodos promedio por árbol  : {np.mean(nodos):.1f} ± {np.std(nodos):.1f}
    Total de nodos (bosque)   : {sum(nodos):,}

    ─── MÉTRICAS DE EVALUACIÓN (conjunto de test 20%) ───────────────────
    Accuracy                  : {accuracy}
    Macro F1-Score            : {macro_f1}
    Weighted F1-Score         : {weighted_f1}
    (Ver classification_report.json para métricas completas por clase)

    ─── ARTEFACTOS SERIALIZADOS ─────────────────────────────────────────
    Modelo  : cicids2017_outputs/random_forest_cicids2017.pkl
    Encoder : cicids2017_outputs/label_encoder_cicids2017.pkl
    """)

    ruta = os.path.join(DIR_TESIS, "model_summary.txt")
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(resumen)
    print(resumen)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 2 — Importancia de características (Top-20)
# ══════════════════════════════════════════════════════════════════════════════

def graficar_feature_importance(clf, feature_names: list):
    print("\n[2/7] Generando gráfico de importancia de características...")

    importancias = pd.Series(clf.feature_importances_, index=feature_names)
    importancias_ord = importancias.sort_values(ascending=False)

    # Exportar tabla completa en CSV
    df_imp = importancias_ord.reset_index()
    df_imp.columns = ['Característica', 'Importancia (Gini)']
    df_imp['Rango'] = range(1, len(df_imp) + 1)
    df_imp['Importancia Acumulada (%)'] = (df_imp['Importancia (Gini)'].cumsum() * 100).round(2)
    ruta_csv = os.path.join(DIR_TESIS, "feature_importance_all.csv")
    df_imp.to_csv(ruta_csv, index=False, encoding='utf-8')
    print(f"  → CSV exportado: {ruta_csv}")

    # ── Top-20 ─────────────────────────────────────────────────────────────
    top20 = importancias_ord.head(20)
    colores = plt.cm.Blues(np.linspace(0.35, 0.9, 20))[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top20.index[::-1], top20.values[::-1],
                   color=colores[::-1], edgecolor='white', linewidth=0.4)

    for bar, val in zip(bars, top20.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', ha='left', fontsize=8.5)

    ax.set_xlabel("Importancia (reducción media de impureza de Gini)", fontsize=11)
    ax.set_title(
        "Top-20 Variables más Importantes — Random Forest CICIDS2017\n"
        "Criterio: Mean Decrease in Impurity (MDI / Gini)",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    ruta = os.path.join(DIR_TESIS, "feature_importance_top20.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")

    return importancias_ord


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 3 — Curva de Pareto acumulativa de importancia
# ══════════════════════════════════════════════════════════════════════════════

def graficar_pareto_importancia(importancias_ord: pd.Series):
    print("\n[3/7] Generando curva de Pareto de importancia acumulada...")

    cumsum  = importancias_ord.cumsum().values * 100
    indices = np.arange(1, len(cumsum) + 1)

    # Puntos clave: cuántas features explican el 80%, 90%, 95%
    puntos = {}
    for umbral in [80, 90, 95]:
        idx = np.searchsorted(cumsum, umbral) + 1
        puntos[umbral] = idx

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(indices, cumsum, color='#2E86AB', linewidth=2.0)
    ax.fill_between(indices, cumsum, alpha=0.15, color='#2E86AB')

    colores_umbrales = {80: '#E84855', 90: '#F4A261', 95: '#3BB273'}
    for umbral, idx in puntos.items():
        ax.axvline(x=idx, color=colores_umbrales[umbral],
                   linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axhline(y=umbral, color=colores_umbrales[umbral],
                   linestyle=':', linewidth=1.0, alpha=0.6)
        ax.annotate(
            f"{umbral}% → {idx} features",
            xy=(idx, umbral),
            xytext=(idx + 1, umbral - 6),
            fontsize=9, color=colores_umbrales[umbral],
            arrowprops=dict(arrowstyle='->', color=colores_umbrales[umbral], lw=1.0)
        )

    ax.set_xlim(1, len(indices))
    ax.set_ylim(0, 102)
    ax.set_xlabel("Número de características (ordenadas por importancia)", fontsize=11)
    ax.set_ylabel("Importancia acumulada (%)", fontsize=11)
    ax.set_title(
        "Curva de Pareto — Importancia Acumulada de Características\n"
        "Random Forest CICIDS2017",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()

    ruta = os.path.join(DIR_TESIS, "feature_importance_cumulative.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")
    for umbral, idx in puntos.items():
        print(f"     {umbral}% de la importancia explicada por {idx} de {len(importancias_ord)} features")


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 4 — Distribución de profundidades reales de los árboles
# ══════════════════════════════════════════════════════════════════════════════

def graficar_tree_depth_distribution(clf):
    print("\n[4/7] Analizando distribución de profundidades del bosque...")

    profundidades = [t.get_depth() for t in clf.estimators_]
    nodos         = [t.tree_.node_count for t in clf.estimators_]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histograma de profundidades
    ax = axes[0]
    ax.hist(profundidades, bins=range(min(profundidades), max(profundidades) + 2),
            color='#2E86AB', edgecolor='white', linewidth=0.5, align='left')
    ax.axvline(np.mean(profundidades), color='#E84855', linestyle='--',
               linewidth=1.5, label=f'Media: {np.mean(profundidades):.1f}')
    ax.set_xlabel("Profundidad real del árbol", fontsize=11)
    ax.set_ylabel("Número de árboles", fontsize=11)
    ax.set_title("Distribución de Profundidades\n(max_depth=10 configurado)", fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Scatter: profundidad vs número de nodos
    ax2 = axes[1]
    ax2.scatter(profundidades, nodos, alpha=0.4, s=18,
                color='#3BB273', edgecolors='none')
    ax2.set_xlabel("Profundidad real del árbol", fontsize=11)
    ax2.set_ylabel("Número de nodos", fontsize=11)
    ax2.set_title("Profundidad vs Número de Nodos\npor árbol individual", fontsize=11, pad=10)
    ax2.grid(linestyle='--', alpha=0.4)

    fig.suptitle(
        f"Análisis Estructural del Bosque — {clf.n_estimators} Árboles",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    ruta = os.path.join(DIR_TESIS, "tree_depth_distribution.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 5 — Visualización de un árbol individual del ensemble
# ══════════════════════════════════════════════════════════════════════════════

def visualizar_arbol_individual(clf, enc: LabelEncoder, feature_names: list,
                                idx_arbol: int = 0, max_depth_plot: int = 4):
    print(f"\n[5/7] Visualizando árbol #{idx_arbol} del ensemble (primeros {max_depth_plot} niveles)...")

    arbol = clf.estimators_[idx_arbol]
    clases_limpias = [_limpiar_nombre(c) for c in enc.classes_]

    fig, ax = plt.subplots(figsize=(26, 10))
    plot_tree(
        arbol,
        feature_names=feature_names,
        class_names=clases_limpias,
        max_depth=max_depth_plot,
        filled=True,
        impurity=True,
        rounded=True,
        proportion=False,
        fontsize=7,
        ax=ax
    )
    ax.set_title(
        f"Árbol de Decisión #{idx_arbol} del Random Forest — Primeros {max_depth_plot} Niveles\n"
        "CICIDS2017 · Clasificación de Tráfico de Red",
        fontsize=12, fontweight='bold', pad=12
    )
    plt.tight_layout()

    ruta = os.path.join(DIR_TESIS, "decision_tree_sample.png")
    fig.savefig(ruta, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 6 — Curvas ROC multiclase (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════════════

def graficar_roc_curves(clf, enc: LabelEncoder, X_test, y_test):
    print("\n[6/7] Calculando curvas ROC multiclase (One-vs-Rest)...")

    clases = enc.classes_
    n_clases = len(clases)
    y_score = clf.predict_proba(X_test)

    # Binarizar etiquetas
    y_test_bin = label_binarize(y_test, classes=list(range(n_clases)))

    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for i in range(n_clases):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # ── Plot ──────────────────────────────────────────────────────────────
    cmap = plt.cm.get_cmap('tab20', n_clases)
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, clase in enumerate(clases):
        clase_limpia = _limpiar_nombre(clase)
        ax.plot(fpr_dict[i], tpr_dict[i],
                color=cmap(i), linewidth=1.4,
                label=f"{clase_limpia}  (AUC = {roc_auc_dict[i]:.3f})")

    # Micro-average en negro
    ax.plot(fpr_micro, tpr_micro, color='black', linewidth=2.2,
            linestyle='--', label=f"Micro-average  (AUC = {auc_micro:.4f})")

    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.8, alpha=0.5, label='Clasificador aleatorio')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=11)
    ax.set_title(
        "Curvas ROC Multiclase — One-vs-Rest (OvR)\n"
        "Random Forest CICIDS2017",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(loc='lower right', fontsize=8, ncol=1, framealpha=0.85)
    ax.grid(linestyle='--', alpha=0.35)
    plt.tight_layout()

    ruta = os.path.join(DIR_TESIS, "roc_curves.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Guardado: {ruta}")

    # Exportar tabla AUC
    df_auc = pd.DataFrame({
        'Clase': [_limpiar_nombre(c) for c in clases],
        'AUC-ROC': [roc_auc_dict[i] for i in range(n_clases)]
    }).sort_values('AUC-ROC', ascending=False)
    df_auc.loc[len(df_auc)] = ['Micro-average', auc_micro]
    ruta_auc = os.path.join(DIR_TESIS, "roc_auc_table.csv")
    df_auc.to_csv(ruta_auc, index=False, encoding='utf-8')
    print(f"  → Tabla AUC exportada: {ruta_auc}")
    print("\n  Tabla AUC-ROC:")
    print(df_auc.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS 7 — Interpretación académica en texto
# ══════════════════════════════════════════════════════════════════════════════

def generar_learning_insights(clf, enc: LabelEncoder, importancias_ord: pd.Series):
    print("\n[7/7] Generando interpretación académica...")

    reporte_json = os.path.join(DIRECTORIO_SALIDA, "classification_report.json")
    clases_problema = []
    if os.path.exists(reporte_json):
        with open(reporte_json, encoding='utf-8') as f:
            rpt = json.load(f)
        excluir = {'accuracy', 'macro avg', 'weighted avg'}
        for clase, metricas in rpt.items():
            if isinstance(metricas, dict) and clase not in excluir:
                if metricas.get('f1-score', 1.0) < 0.5:
                    clases_problema.append(
                        f"  - {_limpiar_nombre(clase)}: F1={metricas['f1-score']:.3f}, "
                        f"Recall={metricas['recall']:.3f}, Support={int(metricas['support'])}"
                    )

    top5 = importancias_ord.head(5)
    top5_str = "\n".join([f"  {i+1}. {feat}  ({val:.4f})"
                          for i, (feat, val) in enumerate(top5.items())])

    insights = textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║        INTERPRETACIÓN ACADÉMICA DEL MODELO — PARA TESIS             ║
    ╚══════════════════════════════════════════════════════════════════════╝

    1. SELECCIÓN DE ALGORITMO
    ─────────────────────────
    Se eligió Random Forest por su capacidad intrínseca de manejar datasets
    altamente desbalanceados como CICIDS2017 (clase BENIGN = ~80% del total)
    sin requerir preprocesamiento adicional de balanceo de clases. El ensemble
    de {clf.n_estimators} árboles independientes reduce la varianza individual de cada
    árbol mediante la técnica de Bagging (Bootstrap Aggregating), mejorando
    la estabilidad predictiva sobre clases minoritarias como Heartbleed o
    Infiltration.

    2. REGULARIZACIÓN Y SESGO-VARIANZA
    ────────────────────────────────────
    El parámetro max_depth=10 actúa como regularizador explícito, limitando
    el sobreajuste que producirían árboles no podados (sin límite de profundidad)
    en un dataset con 504,000+ instancias y 77 características. La selección
    aleatoria de √n_features en cada split (max_features='sqrt' → ~9 features)
    introduce diversidad entre árboles, reduciendo la correlación intra-ensemble.

    3. TOP-5 CARACTERÍSTICAS MÁS DISCRIMINANTES
    ─────────────────────────────────────────────
    Las siguientes variables NetFlow presentan la mayor capacidad discriminante
    según la reducción media de impureza de Gini (MDI):

{top5_str}

    Estas características representan patrones temporales y de volumen en los
    flujos de red, coherentes con la literatura de detección de intrusiones
    basada en flujos (Lashkari et al., 2017 — CICIDS2017 original paper).

    4. CLASES CON BAJO RENDIMIENTO (F1 < 0.5)
    ───────────────────────────────────────────
    Las siguientes clases presentan bajo F1-Score, explicable por su extrema
    subrepresentación en el dataset (problema de long-tail en ciberseguridad):

{chr(10).join(clases_problema) if clases_problema else "  Ninguna clase con F1 < 0.5"}

    Estrategias recomendadas para trabajo futuro:
    - SMOTE / ADASYN para oversampling sintético de clases minoritarias
    - Ajuste de class_weight='balanced' en el clasificador
    - Umbralización adaptativa por clase en la probabilidad de predicción

    5. REPRODUCIBILIDAD
    ────────────────────
    El parámetro random_state=42 garantiza la reproducibilidad completa del
    modelo. La semilla afecta: (a) el muestreo bootstrap de cada árbol,
    (b) la selección de features en cada split, y (c) la partición train/test.
    Cualquier investigador puede reproducir exactamente estos resultados
    usando el archivo .pkl serializado con joblib.

    6. LIMITACIONES
    ───────────────
    - El modelo opera sobre características de flujo pre-computadas (no raw PCAP),
      lo que limita la detección de ataques con patrones de paquetes individuales.
    - max_depth=10 puede ser insuficiente para separar ataques similares en el
      espacio de features (ej. variantes de DoS).
    - Las métricas evaluadas sobre el 20% de test del mismo dataset no garantizan
      generalización a tráfico de redes distintas (overfitting de dominio).
    """)

    ruta = os.path.join(DIR_TESIS, "learning_insights.txt")
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(insights)
    print(insights)
    print(f"  → Guardado: {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  ANÁLISIS DE MODELO PARA TESIS — CICIDS2017 RANDOM FOREST")
    print("=" * 70)

    # 1. Cargar artefactos
    clf, enc = cargar_artefactos()
    feature_names = cargar_feature_names()

    # 2. Ficha técnica
    generar_model_summary(clf, enc, feature_names)

    # 3. Importancia de features
    importancias_ord = graficar_feature_importance(clf, feature_names)

    # 4. Curva de Pareto
    graficar_pareto_importancia(importancias_ord)

    # 5. Distribución de profundidades
    graficar_tree_depth_distribution(clf)

    # 6. Árbol individual
    visualizar_arbol_individual(clf, enc, feature_names,
                                idx_arbol=0, max_depth_plot=4)

    # 7. Curvas ROC — requiere el dataset (puede tardar ~5-10 min)
    print("\n[6/7] Cargando dataset de test para curvas ROC...")
    print("      (Este paso puede tardar varios minutos)")
    try:
        X_test, y_test = cargar_dataset_test(enc, feature_names)
        graficar_roc_curves(clf, enc, X_test, y_test)
    except Exception as e:
        print(f"  [Aviso] No se pudieron generar curvas ROC: {e}")
        print("  Asegúrate de que los CSVs estén en cicids2017_local_data/")

    # 8. Interpretación académica
    generar_learning_insights(clf, enc, importancias_ord)

    print("\n" + "=" * 70)
    print("  Todos los artefactos guardados en:", os.path.abspath(DIR_TESIS))
    print("=" * 70)
    print("\n  Archivos generados:")
    for f in sorted(os.listdir(DIR_TESIS)):
        ruta = os.path.join(DIR_TESIS, f)
        size_kb = os.path.getsize(ruta) / 1024
        print(f"    {f:<45} {size_kb:>8.1f} KB")
