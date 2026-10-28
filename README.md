# Analizador automático de cuadrícula 3×3

Aplicación Streamlit que busca automáticamente dos cortes verticales y dos horizontales para maximizar la desviación estándar de luminancia de la celda central respecto al promedio de las ocho celdas periféricas.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar

Sube `app.py` y `requirements.txt` a un repositorio de GitHub. En Streamlit Community Cloud, crea una app nueva que use `app.py` como archivo principal.

## Criterio

- Convierte RGB a una medida de luminancia aproximada.
- Evalúa posiciones candidatas entre el 10 % y el 90 % de cada dimensión, en pasos del 5 %.
- Requiere que cada una de las tres franjas tenga al menos el 10 % de la dimensión.
- Maximiza: desviación estándar del centro menos promedio de desviaciones estándar de las ocho celdas periféricas.
- Usa tablas integrales de la imagen y su cuadrado para calcular las desviaciones eficientemente.
