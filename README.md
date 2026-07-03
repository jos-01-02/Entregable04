# Suite Web de Analisis CICIDS2017 para Entorno Local

## Guia rapida de recreacion
Para reconstruir el entorno completo desde cero, ejecutar los siguientes pasos en orden:

1. Descargar los CSV del dataset en local:

```bash
node prepare_dataset.mjs
```

2. Generar el reporte de clasificacion y artefactos del modelo:

```bash
c:/Users/josue/source/repos/Entregable-4/.venv/Scripts/python.exe "./cicids2017_pipeline_local (1).py"
```

3. Iniciar el servidor estatico:

```bash
npx serve . -l 3000
```

4. Abrir http://localhost:3000, pulsar Procesar CSV locales y, de forma opcional, cargar [cicids2017_outputs/classification_report.json](cicids2017_outputs/classification_report.json) para habilitar Chart 3 y Chart 4.

## Resumen ejecutivo
Este proyecto integra, en una aplicacion web estatica, las funcionalidades principales de preparacion de datos, analisis exploratorio y visualizacion de resultados sobre el dataset CICIDS2017. La propuesta nace de una transformacion de scripts Python a una interfaz de uso directo en navegador, con enfasis en reproducibilidad local, trazabilidad del procesamiento y estabilidad frente a grandes volumenes de datos.

La solucion fue disenada para ejecucion local con servidor estatico y procesamiento incremental de archivos CSV, evitando dependencias de backend para la etapa de visualizacion. Como resultado, el usuario puede obtener indicadores de distribucion de clases y patrones de puertos de destino, asi como incorporar metricas de modelo desde un reporte de clasificacion exportado previamente.

## Objetivo del proyecto
El objetivo es proporcionar una herramienta de analisis util para documentacion academica y validacion tecnica, que permita:

1. Procesar localmente los archivos CSV de CICIDS2017 sin carga manual archivo por archivo.
2. Generar visualizaciones comparables con el flujo analitico de los scripts base.
3. Presentar resultados de manera clara para informe de tesis.
4. Mantener un flujo reproducible con pasos de ejecucion simples.

## Alcance funcional
La aplicacion implementa los siguientes componentes:

1. Procesamiento local de CSV en streaming para mitigar consumo excesivo de memoria.
2. Resumen de volumen de registros validos, numero de clases y cardinalidad de puertos.
3. Chart 1: mapa de calor de frecuencia entre clases de trafico y Top-20 puertos de destino.
4. Chart 2: distribucion de clases en escala logaritmica.
5. Chart 3: matriz de confusion estimada a partir de recall por clase en classification_report.json.
6. Chart 4: barras comparativas de precision, recall y F1-score por clase.
7. Barra de progreso con estado por archivo y avance global del procesamiento.

## Arquitectura de la solucion
La arquitectura se compone de un frontend estatico y un script auxiliar de descarga:

1. Interfaz y estructura: [index.html](index.html)
2. Estilos visuales: [styles.css](styles.css)
3. Logica de procesamiento y graficos: [app.js](app.js)
4. Descarga automatizada del dataset local: [prepare_dataset.mjs](prepare_dataset.mjs)
5. Scripts base de entrenamiento y analisis en Python:
[analyze_model (1).py](analyze_model%20(1).py), [cicids2017_pipeline_local (1).py](cicids2017_pipeline_local%20(1).py), [generate_charts (1).py](generate_charts%20(1).py)

## Fundamento metodologico
La implementacion web sigue un enfoque de procesamiento incremental y agregacion estadistica:

1. Se leen los CSV desde [cicids2017_local_data](cicids2017_local_data) mediante parseo por bloques.
2. Se normalizan encabezados para resolver variaciones de espacios en columnas.
3. Se validan registros con presencia de Label y Destination Port numerico.
4. Se acumulan contadores por clase y por puerto para evitar mantener todo el dataset en memoria.
5. Se filtran los Top-20 puertos mas frecuentes para construir la matriz de Chart 1.
6. Se renderizan visualizaciones con Plotly y Chart.js en la capa de presentacion.

Este enfoque permite ejecutar analisis de gran tamano con mayor robustez en entorno navegador.

## Requisitos del entorno
1. Sistema operativo Windows, Linux o macOS.
2. Node.js instalado.
3. Conexion a internet para la descarga inicial de los CSV y librerias CDN.
4. Navegador moderno.

## Estructura esperada del proyecto
1. Aplicacion web: [index.html](index.html), [styles.css](styles.css), [app.js](app.js)
2. Descarga de datos: [prepare_dataset.mjs](prepare_dataset.mjs)
3. Carpeta de datos locales: [cicids2017_local_data](cicids2017_local_data)
4. Salidas de pipeline Python: [cicids2017_outputs](cicids2017_outputs)

## Procedimiento de ejecucion recomendado (detalle)
### 1. Descargar dataset en local
Desde la carpeta raiz del proyecto, ejecutar:

```bash
node prepare_dataset.mjs
```

Resultado esperado: creacion o actualizacion de archivos CSV en [cicids2017_local_data](cicids2017_local_data).

### 2. Generar reporte de clasificacion del modelo
Para habilitar los graficos de metricas del modelo, ejecutar el pipeline Python:

```bash
c:/Users/josue/source/repos/Entregable-4/.venv/Scripts/python.exe "./cicids2017_pipeline_local (1).py"
```

Resultado esperado: archivo [cicids2017_outputs/classification_report.json](cicids2017_outputs/classification_report.json) y demas artefactos de salida.

### 3. Iniciar servidor estatico

```bash
npx serve . -l 3000
```

Abrir en navegador: http://localhost:3000

### 4. Ejecutar analisis en la interfaz
1. Pulsar Procesar CSV locales.
2. Esperar el avance de la barra de progreso.
3. Opcionalmente, cargar classification_report.json en el selector de archivo para habilitar Chart 3 y Chart 4.

## Interpretacion de resultados
### Chart 1: Puertos x ataque
Visualiza la intensidad de frecuencias entre clases de trafico y puertos de destino priorizados. Es util para identificar concentraciones de actividad asociadas a perfiles de ataque.

### Chart 2: Distribucion de clases
Presenta la proporcion de instancias por clase en escala logaritmica, permitiendo observar el desbalance estructural del dataset.

### Chart 3: Matriz de confusion estimada
Aproxima la matriz normalizada a partir del recall por clase cuando se provee el reporte JSON. Es una vista sintetica de desempeno por categoria.

### Chart 4: Precision, Recall y F1-score
Compara las tres metricas por clase para detectar fortalezas y debilidades del clasificador en categorias minoritarias y mayoritarias.

## Consideraciones tecnicas para tesis
1. Reproducibilidad: el flujo de ejecucion esta definido en pasos deterministas y rutas locales.
2. Escalabilidad local: el uso de streaming evita sobrecarga de memoria por carga masiva de registros.
3. Trazabilidad: el panel de log documenta eventos clave de lectura y procesamiento por archivo.
4. Transparencia metodologica: la separacion entre etapa de entrenamiento Python y etapa de visualizacion web facilita la auditoria del proceso.

## Limitaciones conocidas
1. El navegador no entrena modelos de scikit-learn.
2. Chart 3 depende de un reporte JSON externo generado por el pipeline Python.
3. La precision de la matriz de confusion estimada esta condicionada a la calidad y estructura del reporte.

## Solucion de problemas
1. Si no aparecen Chart 1 y Chart 2:
Verificar que existen CSV en [cicids2017_local_data](cicids2017_local_data) y ejecutar nuevamente node prepare_dataset.mjs.

2. Si un puerto esta ocupado:
Liberar el puerto o iniciar serve en otro puerto disponible.

3. Si el navegador muestra errores de extensiones:
Probar en ventana privada o con extensiones desactivadas para descartar interferencias externas.

4. Si no aparecen Chart 3 y Chart 4:
Confirmar la existencia de [cicids2017_outputs/classification_report.json](cicids2017_outputs/classification_report.json) y cargarlo manualmente en la interfaz.

## Contribucion academica
La solucion propuesta aporta un puente practico entre analisis de ciberseguridad basado en scripts y comunicacion visual de resultados para escenarios de defensa, auditoria y documentacion cientifica. Su principal valor para tesis radica en la combinacion de reproducibilidad, claridad de resultados y facilidad de demostracion en entorno local controlado.

## Licencia y uso
Este repositorio se orienta a fines academicos y de investigacion. Se recomienda citar la fuente del dataset CICIDS2017 y documentar las condiciones de uso conforme a lineamientos institucionales.
