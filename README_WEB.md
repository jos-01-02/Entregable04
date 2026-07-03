# CICIDS2017 Web Suite

Esta version web integra y complementa la funcionalidad de:

- analyze_model (1).py
- cicids2017_pipeline_local (1).py
- generate_charts (1).py

## Que hace en modo web

- Lee CSV locales desde `cicids2017_local_data` para evitar CORS en navegador.
- Limpia datos basicos (filas invalidas y deduplicacion ligera).
- Genera:
  - Chart 1: heatmap de puertos destino vs clases.
  - Chart 2: distribucion de clases (escala log).
  - Chart 3: matriz de confusion estimada desde `classification_report.json`.
  - Chart 4: precision, recall y F1 por clase.

Nota: al ejecutarse en `npx serve` (servidor estatico), no se entrena RandomForest ni se cargan archivos `.pkl`.

## Ejecutar con npx serve

Desde la carpeta del proyecto:

```bash
node prepare_dataset.mjs
npx serve .
```

Abre la URL local mostrada por `serve` y entra a `index.html`.

## Archivos web

- `index.html`
- `styles.css`
- `app.js`
- `prepare_dataset.mjs`

## Flujo recomendado

1. Ejecuta tu pipeline en Python para obtener `classification_report.json`.
2. Ejecuta `node prepare_dataset.mjs` para descargar los CSV desde repositorios publicos.
3. Inicia `npx serve .`.
4. En la pagina, pulsa el boton para procesar CSV locales.
5. (Opcional) Carga `classification_report.json` para habilitar Charts 3 y 4.
