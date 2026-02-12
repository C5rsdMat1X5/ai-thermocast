# 🌡️ AI ThermoCast - Climate Forecaster

**AI ThermoCast** es una aplicación web interactiva que utiliza Redes Neuronales Artificiales (ANN) para predecir anomalías de temperatura global basadas en diferentes escenarios de emisiones de CO2.

La aplicación permite entrenar modelos en tiempo real, simular futuros climáticos mediante el ajuste de sectores industriales (Energía, Transporte, Agricultura, etc.) y visualizar los resultados en gráficos dinámicos.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![NumPy](https://img.shields.io/badge/NumPy-Computation-orange)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualization-pink)

## ✨ Características Principales

- **🧠 Entrenamiento en Tiempo Real:** Entrena una red neuronal personalizada desde el navegador, visualizando el estado del proceso.
- **🔄 Dos Modos de Operación:**
  - **Modo Simple:** Proyección basada en tendencias globales (escenarios: Conservador, Optimista, Pesimista, etc.).
  - **Modo Avanzado:** Control granular reduciendo o aumentando emisiones por sectores específicos (Energía, Industria, Transporte, Edificios, Agricultura).
- **📂 Gestión de Modelos:**
  - Descarga tu modelo entrenado (`.npz`) para usarlo después.
  - **Drag & Drop:** Carga modelos pre-entrenados arrastrándolos a la interfaz.
- **📊 Visualización Interactiva:** Gráficos dinámicos con Chart.js que muestran la trayectoria de temperatura proyectada hasta 1000 años.
- **💾 Exportación de Datos:** Descarga las predicciones generadas en formato CSV.
- **🎨 Diseño Moderno:** Interfaz de usuario estilo "Glassmorphism" con animaciones fluidas y totalmente responsiva.

## 🛠️ Tecnologías Utilizadas

- **Backend:** Python 3, Flask.
- **IA / Matemáticas:** NumPy (Red neuronal implementada desde cero con matrices), Pandas.
- **Frontend:** HTML5, CSS3 (Variables, Flexbox/Grid, Glassmorphism), JavaScript.
- **Gráficos:** Chart.js.

## 🚀 Instalación y Uso

Sigue estos pasos para ejecutar el proyecto en tu máquina local:

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ai-thermocast.git
cd ai-thermocast
```

### 2. Crear un entorno virtual (Recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

Asegúrate de tener un archivo `requirements.txt` o instala las librerías manualmente:

```bash
pip install flask numpy pandas
```

### 4. Ejecutar la aplicación

```bash
python app.py
```

### 5. Abrir en el navegador

Visita la siguiente dirección en tu navegador web:
`http://localhost:3434`

## 📂 Estructura del Proyecto

```text
ai-thermocast/
│
├── app.py              # Punto de entrada de la aplicación Flask
├── model.py            # Clase de la Red Neuronal (Network)
├── model_manager.py    # Singleton para gestionar el estado del entrenamiento
├── predictor.py        # Lógica de simulación y escenarios futuros
├── data_utils.py       # Carga y procesamiento de datasets (CSV)
│
├── static/
│   ├── styles.css      # Estilos CSS (Glassmorphism)
│   ├── favicon.ico     # Icono de la web
│   └── downloads/      # Carpeta temporal para .npz y .csv generados
│
├── templates/
│   └── index.html      # Interfaz principal (HTML + Jinja2)
│
└── README.md           # Documentación del proyecto
```

## 🎮 Guía de Uso

1.  **Cargar o Entrenar:**
    - Si tienes un archivo `.npz`, arrástralo a la zona de carga superior izquierda.
    - Si no, ve a la tarjeta **"Entrenamiento"**, selecciona el modo (Simple o Avanzado), ajusta el _Learning Rate_ y los _Pasos_, y dale a "Iniciar".
2.  **Predecir:**
    - Una vez listo el modelo, ve a la tarjeta **"Predicción"**.
    - Elige cuántos años quieres proyectar.
    - Si estás en **Modo Simple**, elige un escenario preestablecido.
    - Si estás en **Modo Avanzado**, ajusta los sliders de cada sector industrial.
3.  **Analizar:**
    - Observa el gráfico generado a la derecha.
    - Descarga el CSV si necesitas los datos brutos.
    - Descarga el modelo `.npz` si quieres guardarlo para después.

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar la precisión del modelo o el diseño:

1.  Haz un Fork del proyecto.
2.  Crea una rama para tu feature (`git checkout -b feature/NuevaMejora`).
3.  Haz Commit de tus cambios (`git commit -m 'Agregada nueva funcionalidad'`).
4.  Haz Push a la rama (`git push origin feature/NuevaMejora`).
5.  Abre un Pull Request.

---

**Desarrollado por:** Matias Henriquez
