# Miniproyecto 2: Clasificación de Sentimientos en Películas (IMDb) con Redes Recurrentes

¡Bienvenido al proyecto! Este trabajo forma parte del curso de **Técnicas de Deep Learning** de la Maestría en Inteligencia Artificial.

---

## 🎬 ¿De qué trata este proyecto? (En lenguaje sencillo)

Imagina que tienes miles de comentarios y reseñas de personas que fueron a ver una película en internet. Algunas reseñas dicen cosas increíbles como *"¡la mejor película del año, actuaciones brillantes!"*, mientras que otras dicen *"un desastre total, perdí dos horas de mi vida"*. 

El objetivo de este proyecto es enseñarle a una computadora (mediante **Inteligencia Artificial con Redes Neuronales Recurrentes**) a leer esos textos y predecir automáticamente si la opinión fue:
* **👍 Positiva (Clase 1):** Al usuario le gustó la película.
* **👎 Negativa (Clase 0):** Al usuario no le gustó la película.

Para lograrlo, comparamos dos tipos de redes especializadas en procesar secuencias de palabras: **LSTM** y **GRU**, evaluando cuál de las dos es más precisa y eficiente.

---

## 📁 ¿Cómo está organizada esta carpeta?

Hemos organizado el proyecto de manera limpia para que todo sea fácil de encontrar:

```text
MiniProyecto2/
│
├── main.ipynb             <- Cuaderno principal de Jupyter donde se corre todo el código y se ven las gráficas.
├── requirements.txt       <- Lista de librerías de Python necesarias para que todo funcione.
├── README.md              <- Este instructivo que estás leyendo.
│
├── archive/               <- Carpeta con los datos:
│   └── movie.csv          <- Archivo con 40,000 reseñas etiquetadas de IMDb.
│
├── Setup/                 <- Herramientas de instalación y código modular:
│   ├── library.py         <- Módulo con todas las funciones (limpieza de texto, modelos, entrenamiento y gráficos).
│   ├── setup_env.bat      <- Script para Windows: instala todo y configura el kernel con un solo clic.
│   ├── setup_env.py       <- Script equivalente en Python para crear el entorno y registrar el kernel.
│   └── requirements.txt   <- Copia de respaldo de dependencias para el instalador.
│
├── Latex/                 <- Carpeta con el informe académico:
│   └── main.tex           <- Documento en formato IEEE (máximo 2 páginas) listo para entregar o subir a Overleaf.
│
└── Documentacion/         <- Guías originales de la maestría:
    ├── microproyecto_RNN.pdf           <- Enunciado y objetivos del proyecto.
    └── 202611_Rubrica_MNP_MAIA_4213.pdf <- Criterios de evaluación del profesor.
```

---

## 🚀 Guía de Instalación Rápida (Paso a Paso)

No necesitas instalar cosas complejas a mano. Puedes dejar todo configurado en un par de minutos siguiendo **cualquiera de estas dos opciones**:

### Opción 1: Automática con un clic (Recomendada en Windows)
1. Entra a la subcarpeta `Setup`.
2. Haz doble clic sobre el archivo **`setup_env.bat`**.
3. Se abrirá una ventana negra que automáticamente:
   * Creará el entorno virtual (`env_miniproyecto2`).
   * Instalará PyTorch, Pandas, Scikit-Learn y Matplotlib.
   * Registrará el kernel en Jupyter con el nombre **`Python (MiniProyecto 2 - RNN)`**.
4. Cuando termine, presiona cualquier tecla y listo.

---

### Opción 2: Desde la terminal de comandos
Si prefieres hacerlo por consola o estás en Mac/Linux:
1. Abre una terminal dentro de la carpeta `MiniProyecto2`.
2. Ejecuta:
   ```bash
   python Setup/setup_env.py
   ```
3. El script se encargará de todo y te avisará cuando el entorno esté listo.

---

## 💻 ¿Cómo abrir y ejecutar el proyecto?

Una vez configurado el entorno:

1. Abre **Visual Studio Code** o tu navegador con **Jupyter Notebook / JupyterLab** en la carpeta `MiniProyecto2`.
2. Abre el archivo **`main.ipynb`**.
3. En la esquina superior derecha de la ventana, asegúrate de seleccionar el kernel:
   👉 **`Python (MiniProyecto 2 - RNN)`** (o el intérprete de Python del entorno creado).
4. Dale clic al botón **"Run All" (Ejecutar todo)** o ve corriendo celda por celda con `Shift + Enter`.

---

## 📊 ¿Qué vas a ver al ejecutarlo?

El cuaderno corre de inicio a fin en unos **2 a 3 minutos** en cualquier computador normal (incluso sin tarjeta gráfica / GPU):

1. **Exploración:** Verás una gráfica de barras que confirma que las opiniones están 50% positivas y 50% negativas.
2. **Preprocesamiento:** Se limpian los textos (quitando etiquetas web como `<br />`, puntuación rara y mayúsculas) y se convierten en números con relleno a la izquierda (*pre-padding*) para que la red entienda el mensaje final de la reseña.
3. **Entrenamiento:** Entrena primero el modelo **LSTM** y luego el modelo **GRU** durante varias épocas. Cuenta con *Early Stopping* (parada inteligente), es decir, si el modelo ya aprendió lo suficiente y no mejora más, se detiene solo para no perder tiempo ni sobreajustarse.
4. **Gráficas de Aprendizaje:** Dos paneles a color mostrando cómo disminuyó el error (*Loss*) y cómo aumentó la exactitud (*Accuracy*).
5. **Tabla de Resultados y Matrices de Confusión:** Una tabla comparativa clara con Exactitud, Precisión, Sensibilidad y F1-Score, acompañada de matrices de confusión visuales.
6. **Casos Reales (Aciertos y Errores):** Muestra ejemplos concretos de reseñas donde la IA acertó con seguridad (como críticas con elogios claros) y ejemplos donde falló (como cuando una persona usa sarcasmo o habla de cine de terror con palabras oscuras que la red confunde).

---

## 👥 Autores
* Jhon Edwuard Zacipa Casas
* José Luis Sarta Alvarez
* Marco Julio Gómez Amado

*Maestría en Inteligencia Artificial — Técnicas de Deep Learning*

