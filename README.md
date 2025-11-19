

# 🎌 Clasificación de Caracteres Japoneses con Redes Neuronales

Este proyecto aborda el desafío de **Reconocimiento Óptico de Caracteres (OCR)** para la identificación de caracteres japoneses antiguos (Kuzushiji). Se implementan y comparan dos enfoques arquitectónicos: una **Red Neuronal construida desde cero (from scratch)** utilizando únicamente NumPy, y una implementación moderna utilizando **PyTorch**.

El objetivo es clasificar correctamente imágenes de $28 \times 28$ píxeles en 49 clases distintas, explorando el impacto de diversos hiperparámetros y técnicas de optimización.

## 🚀 Características del Proyecto

El sistema no es solo una "caja negra", sino una exploración profunda de los fundamentos del Deep Learning. Incluye:

  * **Implementación "From Scratch" (`numpy`):**
      * Arquitectura MLP configurable (capas ocultas dinámicas).
      * Backpropagation manual.
      * Funciones de activación (ReLU, Softmax).
      * Optimizador **Adam** implementado manualmente.
      * **Rate Scheduling** (Lineal con saturación y Exponencial).
      * Regularización L2 y Early Stopping.
  * **Implementación PyTorch:**
      * Réplica de la arquitectura para validación de resultados.
      * Uso de `torch.nn` y `torch.optim`.
  * **Análisis de Modelos:** Comparativa entre 5 modelos distintos (m0 a m4) variando desde configuraciones básicas hasta redes con overfitting forzado.

## 🛠️ Tecnologías Utilizadas

  * **Python**: Lenguaje principal.
  * **NumPy**: Cálculos matriciales y álgebra lineal para la red "from scratch".
  * **PyTorch**: Framework de Deep Learning para los modelos avanzados (m2, m3, m4).
  * **Matplotlib**: Visualización de curvas de pérdida (loss) y precisión (accuracy).
  * **Pandas**: Manejo de datos y generación de archivos de salida.

## 📊 Resultados Destacados

Basado en el informe técnico adjunto, se evaluaron diferentes configuraciones:

| Modelo | Tipo | Configuración | Accuracy (Val) | Observación |
| :--- | :--- | :--- | :--- | :--- |
| **m0** | Scratch | Básico, SGD | \~56% | Baseline |
| **m1** | Scratch | Adam, L2, Early Stopping | \~63% | Mejoras significativas por optimizador |
| **m2** | PyTorch | Réplica de m1 | \~63% | Validación de la implementación manual |
| **m3** | PyTorch | Grid Search [100, 80] | **\~63%** | Modelo seleccionado por mejor generalización |
| **m4** | PyTorch | Overfitting [5000] | \~70% | Alta varianza, poca generalización |

El modelo **m3** fue seleccionado para la inferencia final debido a su equilibrio entre precisión y capacidad de generalización, evitando el overfitting observado en redes masivas (m4).

## 👤 Autor

**Manuel Borrell**
*Ingeniería en Inteligencia Artificial - Universidad de San Andrés*

-----

*Este proyecto fue desarrollado como parte del curso de Aprendizaje Automático y Aprendizaje Profundo.*
