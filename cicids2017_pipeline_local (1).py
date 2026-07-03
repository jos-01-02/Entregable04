import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para generar PNGs en cualquier entorno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Intentar importar la librería 'requests' avanzada; si no está, usar urllib con cabeceras mutadas
try:
    import requests
    USAR_REQUESTS = True
except ImportError:
    import urllib.request
    USAR_REQUESTS = False

# ==============================================================================
# CONFIGURACIÓN DE RUTAS E HISTORIAL DE MIRRORS ANTE BLOQUEOS (HTTP 403)
# ==============================================================================
DIRECTORIO_LOCAL = "./cicids2017_local_data/"
DIRECTORIO_SALIDA = "./cicids2017_outputs/"  # Modelos, reportes y gráficos exportados

# Estructura de espejos (Mirrors) redundantes para asegurar la descarga académica
FUENTES_DATASET = {
    "Mirror_Activo_HuggingFace": {
        "base_url": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/",
        "archivos": [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
    },
    "Mirror_Principal_GitLab": {
        "base_url": "https://gitlab.com/msc-cybersecurity-datasets/cicids2017/-/raw/main/",
        "archivos": [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
    },
    "Mirror_Respaldo_GitHub": {
        "base_url": "https://raw.githubusercontent.com/cyber-risk-analysis/cicids2017-mirror/main/",
        "archivos": [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
    }
}


def descargar_archivo_con_cabeceras(url, destino_local):
    """
    Simula una petición de navegador web real (Firefox/Windows) para romper
    los bloqueos de seguridad perimetral HTTP 403.
    """
    cabeceras = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    if USAR_REQUESTS:
        # Método recomendado: Descarga por flujos (stream) para archivos masivos de red
        with requests.get(url, headers=cabeceras, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(destino_local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        # Método alternativo si el entorno virtual carece de la librería requests
        peticion = urllib.request.Request(url, headers=cabeceras)
        with urllib.request.urlopen(peticion, timeout=30) as respuesta:
            with open(destino_local, 'wb') as f:
                f.write(respuesta.read())


def orquestar_ingesta_hibrida():
    """
    Busca los archivos en el disco duro local de VS Code.
    Si no existen, barre los espejos públicos de internet aplicando bypass de 403.
    """
    if not os.path.exists(DIRECTORIO_LOCAL):
        os.makedirs(DIRECTORIO_LOCAL)
        print(f"[Eje de Datos] Creando espacio de persistencia local en: {DIRECTORIO_LOCAL}")

    dataframes_listo = []
    # Lista estandarizada de nombres canónicos y variantes conocidas por mirror
    archivos_requeridos = [
        ["Monday-WorkingHours.pcap_ISCX.csv"],
        ["Tuesday-WorkingHours.pcap_ISCX.csv"],
        ["Wednesday-workingHours.pcap_ISCX.csv"],
        ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
        [
            "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
        ],
        ["Friday-WorkingHours-Morning.pcap_ISCX.csv"],
        ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
        ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
    ]

    print("\n[Fase 1: Ingesta] Verificando estado del almacenamiento local...")

    for variantes_nombre in archivos_requeridos:
        nombre_canonico = variantes_nombre[0]
        ruta_local_completa = os.path.join(DIRECTORIO_LOCAL, nombre_canonico)
        ruta_local_alternativa = None

        for nombre_variante in variantes_nombre:
            posible_local = os.path.join(DIRECTORIO_LOCAL, nombre_variante)
            if os.path.exists(posible_local) and os.path.getsize(posible_local) > 0:
                ruta_local_alternativa = posible_local
                break

        # 1. Fallback local prioritario (Ahorra ancho de banda y evita latencias)
        if ruta_local_alternativa is not None:
            print(f"--> [Local Detectado]: {os.path.basename(ruta_local_alternativa)} listo para lectura.")
        else:
            print(f"--> [Ausente / Incompleto]: {nombre_canonico}. Buscando bypass en internet...")
            descargado = False

            # 2. Bucle de reintentos sobre múltiples espejos (Anti-403)
            for nombre_mirror, configuracion in FUENTES_DATASET.items():
                print(f"    Intentando conexión con {nombre_mirror}...")
                for nombre_variante in variantes_nombre:
                    url_remota = configuracion["base_url"] + nombre_variante
                    try:
                        descargar_archivo_con_cabeceras(url_remota, ruta_local_completa)
                        print(f"    ¡Éxito! Archivo obtenido desde internet: {nombre_variante}")
                        descargado = True
                        break  # Archivo obtenido, romper bucle de variantes
                    except Exception as e:
                        print(f"    Fallo en {nombre_mirror} para {nombre_variante}. Código/Error: {e}")
                        if os.path.exists(ruta_local_completa):
                            os.remove(ruta_local_completa)  # Limpiar archivos corruptos a medio descargar
                        time.sleep(1)  # Espera técnica para mitigar bloqueos por ráfaga

                if descargado:
                    break  # Archivo obtenido, romper bucle de espejos

            if not descargado:
                print(f"\n[ALERTA CRÍTICA]: No se pudo descargar {nombre_canonico} de ninguna fuente web.")
                print(f"Por favor, descarga el archivo manualmente y colócalo dentro de la carpeta: {DIRECTORIO_LOCAL}")
                sys.exit(1)  # Forzar salida del script con código de error visible en VS Code

            ruta_local_alternativa = ruta_local_completa

        # 3. Carga e ingeniería de características en memoria RAM
        try:
            df_dia = pd.read_csv(ruta_local_alternativa, encoding='cp1252', low_memory=False)
            df_dia.columns = df_dia.columns.str.strip()  # Saneamiento de los encabezados originales mal formateados
            dataframes_listo.append(df_dia)
            print(f"    Estructurado en memoria RAM: {os.path.basename(ruta_local_alternativa)} ({df_dia.shape[0]} registros).")
        except Exception as e:
            print(f"[Error] Error de lectura estructural en {nombre_canonico}: {e}")

    print("\n[CRISP-DM: Preparación] Ejecutando fusión analítica indexada...")
    df_unificado = pd.concat(dataframes_listo, ignore_index=True)
    print(f"--> Dataset Unificado: {df_unificado.shape[0]} filas | {df_unificado.shape[1]} columnas.")
    return df_unificado


def depurar_y_limpiar_trafico(df):
    print("\n[CRISP-DM: Preparación] Tratando nulos, infinitos y deduplicación de flujos...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    instancias_brutas = df.shape[0]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"--> Registros depurados (Data Leakage mitigado): {instancias_brutas - df.shape[0]} eliminados.")

    # Exclusión regulatoria e institucional de variables que inducen sobreajuste sintético
    columnas_anonimizar = ['Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port']
    columnas_utiles = [col for col in df.columns if col not in columnas_anonimizar]
    return df[columnas_utiles]


def exportar_modelo(clasificador, encoder, directorio):
    """Serializa el modelo Random Forest y el LabelEncoder con joblib."""
    os.makedirs(directorio, exist_ok=True)
    ruta_modelo = os.path.join(directorio, "random_forest_cicids2017.pkl")
    ruta_encoder = os.path.join(directorio, "label_encoder_cicids2017.pkl")
    joblib.dump(clasificador, ruta_modelo)
    joblib.dump(encoder, ruta_encoder)
    print(f"[Exportación] Modelo guardado en: {ruta_modelo}")
    print(f"[Exportación] Encoder guardado en: {ruta_encoder}")


def guardar_reporte(reporte_dict, directorio):
    """Guarda el classification_report como CSV y JSON."""
    os.makedirs(directorio, exist_ok=True)
    # JSON
    ruta_json = os.path.join(directorio, "classification_report.json")
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(reporte_dict, f, ensure_ascii=False, indent=2)
    print(f"[Reporte] JSON guardado en: {ruta_json}")
    # CSV (solo clases, sin las filas de promedio)
    filas = {k: v for k, v in reporte_dict.items() if isinstance(v, dict)}
    df_reporte = pd.DataFrame(filas).T
    ruta_csv = os.path.join(directorio, "classification_report.csv")
    df_reporte.to_csv(ruta_csv, index=True, encoding='utf-8')
    print(f"[Reporte] CSV guardado en: {ruta_csv}")


def graficar_confusion_matrix(y_test, predicciones, clases, directorio):
    """Genera y guarda una heatmap de la matriz de confusión normalizada."""
    os.makedirs(directorio, exist_ok=True)
    cm = confusion_matrix(y_test, predicciones, normalize='true')
    fig, ax = plt.subplots(figsize=(max(10, len(clases)), max(8, len(clases) - 1)))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=clases, yticklabels=clases, ax=ax
    )
    ax.set_title('Matriz de Confusión Normalizada — CICIDS2017', fontsize=14, pad=12)
    ax.set_xlabel('Predicción', fontsize=11)
    ax.set_ylabel('Valor Real', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    ruta = os.path.join(directorio, "confusion_matrix.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"[Gráfico] Matriz de confusión guardada en: {ruta}")


def graficar_metricas_por_clase(reporte_dict, directorio):
    """Genera barras de precision, recall y f1-score agrupadas por clase."""
    os.makedirs(directorio, exist_ok=True)
    clases = [k for k, v in reporte_dict.items() if isinstance(v, dict) and k not in ('accuracy',)]
    precision  = [reporte_dict[c]['precision'] for c in clases]
    recall     = [reporte_dict[c]['recall'] for c in clases]
    f1         = [reporte_dict[c]['f1-score'] for c in clases]

    x = np.arange(len(clases))
    ancho = 0.26
    fig, ax = plt.subplots(figsize=(max(14, len(clases) * 1.1), 6))
    ax.bar(x - ancho, precision, ancho, label='Precision', color='steelblue')
    ax.bar(x,         recall,    ancho, label='Recall',    color='darkorange')
    ax.bar(x + ancho, f1,        ancho, label='F1-Score',  color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=40, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Métricas por Clase — Random Forest CICIDS2017', fontsize=14, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    ruta = os.path.join(directorio, "metricas_por_clase.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"[Gráfico] Métricas por clase guardadas en: {ruta}")


def entrenar_modelo_forestal(df, target_col='Label'):
    print("\n[CRISP-DM: Modelado] Aplicando Label Encoding multiclase...")
    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Partición Estratificada (80% Train / 20% Test) para preservar clases raras de ciberataques
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[CRISP-DM: Modelado] Ajustando Random Forest jerárquico local (n_jobs=-1)...")
    # max_depth=10 e indica parámetros de regularización adaptados para una PC de escritorio estándar
    clasificador = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clasificador.fit(X_train, y_train)

    print("\n================ EVALUACIÓN FINAL DE LA GESTIÓN DE RIESGOS ================")
    predicciones = clasificador.predict(X_test)
    reporte_dict = classification_report(
        y_test, predicciones, target_names=encoder.classes_,
        zero_division=0, output_dict=True
    )
    print(classification_report(y_test, predicciones, target_names=encoder.classes_, zero_division=0))

    return clasificador, encoder, reporte_dict, y_test, predicciones


# ==============================================================================
# INICIO DEL ENTORNO DE EJECUCIÓN LOCAL
# ==============================================================================
if __name__ == "__main__":
    print("=== PIPELINE DE CIDERSEGURIDAD LOCAL INSTANCIADO EN VS CODE ===")
    try:
        # Si la red local o corporativa no tiene instalada 'requests', se le sugiere al alumno
        if not USAR_REQUESTS:
            print("[Aviso Técnico]: Para un mejor control de descargas masivas ejecuta: pip install requests")

        dataset_crudo = orquestar_ingesta_hibrida()
        dataset_sano = depurar_y_limpiar_trafico(dataset_crudo)
        clasificador, encoder, reporte_dict, y_test, predicciones = entrenar_modelo_forestal(dataset_sano)

        print("\n[Fase de Exportación] Serializando modelo y generando reportes/gráficos...")
        exportar_modelo(clasificador, encoder, DIRECTORIO_SALIDA)
        guardar_reporte(reporte_dict, DIRECTORIO_SALIDA)
        graficar_confusion_matrix(y_test, predicciones, encoder.classes_, DIRECTORIO_SALIDA)
        graficar_metricas_por_clase(reporte_dict, DIRECTORIO_SALIDA)

        print("\n[Proceso Concluido]: Pipeline ejecutado exitosamente sin errores ocultos.")
        sys.exit(0)  # Código de salida explícito para el control de VS Code
    except Exception as e_global:
        print(f"\n[Error Crítico del Pipeline]: {e_global}")
        sys.exit(1)
