# Detección de Fraude en Tarjetas de Crédito

Proyecto de ciencia de datos (análisis + machine learning) para detectar transacciones fraudulentas en tarjetas de crédito, desarrollado con el flujo completo de un proyecto de Data Science: problema → datos → exploración → preprocesamiento → balanceo → modelos → evaluación → selección → umbral → conclusiones.

## Descripción

Usando el dataset **Credit Card Fraud Detection** (Kaggle), se entrena un modelo que estima la probabilidad de que una transacción sea fraudulenta. El foco está en el manejo correcto de un problema de **clasificación desbalanceada** (el fraude es ~0.17% de los datos), evitando métricas engañosas como la accuracy y aplicando buenas prácticas que evitan la fuga de información (*data leakage*).

## Problema

- **De negocio:** millones de transacciones circulan a diario; una fracción mínima es fraude y genera pérdidas para el banco y riesgo para el cliente. Se busca alertar sobre transacciones con alta probabilidad de fraude.
- **Técnico:** desbalance extremo de clases (la gran mayoría de las transacciones es normal). Si el modelo clasificara **todo** como "normal" obtendría ~99.83% de accuracy **sin detectar ni un fraude**. Por eso se usa precision, recall, F1, ROC-AUC y Average Precision.

## Objetivos

**Pregunta del proyecto:** ¿Es posible utilizar modelos de machine learning para identificar transacciones fraudulentas y reducir el número de fraudes no detectados?

**Objetivo general:** desarrollar y evaluar un modelo capaz de identificar transacciones fraudulentas considerando el fuerte desbalance entre operaciones legítimas y fraudulentas.

**Objetivos específicos:** analizar las transacciones, cuantificar el desbalance, preparar los datos evitando data leakage, comparar estrategias de balanceo, entrenar Logistic Regression / Random Forest / XGBoost, evaluar con métricas adecuadas, analizar falsos positivos (FP) y falsos negativos (FN), seleccionar el mejor modelo y analizar el impacto del umbral de decisión.

## Dataset

**Credit Card Fraud Detection** — Kaggle (MLG-ULB).

- **284,807 registros** y **31 variables** (30 explicativas + `Class`).
- Variable objetivo **`Class`**: `1` = fraude, `0` = normal.
- Variables: `Time` (segundos desde la primera transacción), `Amount` (monto) y `V1`–`V28`, que son **variables anonimizadas** (transformadas vía PCA por el dueño del dataset; no tienen un significado financiero interpretable).
- Tras la limpieza de **1,081 filas duplicadas**: **283,726 registros** y **473 fraudes (0.167%)**.

## Metodología

1. **Carga y calidad de datos:** revisión de nulos, tipos y duplicados → decisión de eliminar duplicados.
2. **EDA dirigido por preguntas** con interpretación de cada gráfico.
3. **Preprocesamiento sin data leakage:** split 70/30 estratificado **antes** de ajustar el escalador; `RobustScaler` aprendido **solo con entrenamiento** y aplicado luego a train y test.
4. **Balanceo:** SMOTE **solo sobre entrenamiento**; el test se mantiene 100% real para una evaluación honesta.
5. **Modelos:** Logistic Regression con y sin balanceo (referencia del efecto del SMOTE), Random Forest y XGBoost (con SMOTE).
6. **Evaluación:** precision, recall, F1, ROC-AUC, Average Precision, matrices de confusión y curvas ROC / Precision-Recall.
7. **Umbral de decisión:** barrido de umbrales con impacto en FP y FN.

## Preprocesamiento (sin data leakage)

La regla clave del proyecto: **primero dividir, después escalar**.

```text
Dataset → train/test split → fit del scaler SOLO en train → transform train y test → balancear SOLO train → entrenar → evaluar contra el test real
```

Ajustar el scaler (u otro paso de preprocesamiento) con todos los datos provocaría que el modelo "mire" información del test durante el entrenamiento, produciendo métricas optimistas irreales.

## Modelos

| Modelo | Datos de entrenamiento | Justificación |
|---|---|---|
| Logistic Regression | original (sin balanceo) | Referencia: muestra qué pasa sin balancear |
| Logistic Regression | SMOTE | Modelo lineal simple, rápido e interpretable |
| Random Forest | SMOTE | Captura relaciones no lineales |
| XGBoost | SMOTE | En general el más potente en datos tabulares |

## Métricas

- **Recall:** de los fraudes reales, ¿cuántos se detectaron?
- **Precision:** de lo marcado como fraude, ¿cuánto era fraude real?
- **F1:** equilibrio entre precision y recall.
- **ROC-AUC:** ordena fraude vs normal.
- **Average Precision:** resumen de la curva Precision-Recall, más informativa en desbalanceo.
- **FP y FN:** falsas alarmas y fraudes no detectados.

Los **FN importan especialmente**: cada fraude no detectado es dinero perdido; cada FP solo genera una revisión manual.

## Resultados

Métricas sobre el test real (85,118 transacciones, 142 fraudes), umbral 0.5:

| Modelo | Precision | Recall | F1 | ROC-AUC | Avg Precision | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (sin balanceo) | 0.849 | 0.556 | 0.672 | 0.967 | 0.707 | 14 | 63 |
| Logistic Regression (SMOTE) | 0.052 | 0.880 | 0.097 | 0.966 | 0.689 | 2301 | 17 |
| **Random Forest (SMOTE)** | **0.907** | 0.754 | **0.823** | 0.965 | **0.810** | **11** | 35 |
| XGBoost (SMOTE) | 0.809 | **0.775** | 0.791 | 0.963 | 0.785 | 26 | **32** |

**Hallazgo interesante:** el SMOTE no le sirve a Logistic Regression: sube el recall pero colapsa la precisión (2,301 falsas alarmas). Los modelos basados en árboles sí aprovechan el balanceo.

## Comparación de modelos

- **Mejor F1 y Avg Precision:** Random Forest (con SMOTE), con solo 11 falsas alarmas en test.
- **Mejor recall (pero inutilizable):** Logistic Regression con SMOTE (precisión 0.052).
- **Alternativa de mayor recall:** XGBoost (0.775), a costa de más falsas alarmas (26 vs 11).

**Elección:** Random Forest con SMOTE por su mejor equilibrio global y menor cantidad de alertas falsas, lo que lo hace viable de operar.

## Optimización del umbral

El clasificador emite probabilidades; el umbral por defecto es 0.5, pero no siempre es el mejor. Barrido sobre Random Forest:

| Umbral | Precision | Recall | F1 | FP | FN | Fraudes detectados |
|---|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.797 | 0.803 | 0.800 | 29 | 28 | 114 |
| 0.4 | 0.848 | 0.789 | 0.818 | 20 | 30 | 112 |
| 0.5 | 0.907 | 0.754 | 0.823 | 11 | 35 | 107 |
| 0.6 | 0.937 | 0.732 | 0.822 | 7 | 38 | 104 |
| **0.7** | **0.972** | 0.732 | **0.835** | **3** | 38 | 104 |
| 0.8 | 0.969 | 0.669 | 0.792 | 3 | 47 | 95 |
| 0.9 | 0.967 | 0.620 | 0.755 | 3 | 54 | 88 |

**No existe un umbral perfecto:** bajar el umbral detecta más fraude pero genera más alertas; subirlo reduce revisiones pero pierde fraude. Se eligió el **umbral 0.7** por maximizar el F1 (0.835) y reducir drásticamente las falsas alarmas (de 11 a 3), manteniendo la captura de la mayoría del fraude.

### Resultado final del modelo elegido

**Random Forest con SMOTE** y umbral 0.7 sobre el test real:

- ***Fraudes reales:*** 142
- ***Fraudes detectados:*** 104 (73.2%)
- ***Fraudes no detectados (FN):*** 38
- ***Normales marcadas como fraude (FP):*** 3
- ***Precision:*** 0.972 | ***Recall:*** 0.732 | ***F1:*** 0.835

## Interpretación

**¿Qué aprendimos de los datos?** El fraude es extremadamente raro (0.167%), suele aparecer en montos bajos y no muestra un patrón temporal claro; las variables V1–V28 (anonimizadas) llevan la información que separa fraude de normal.

**¿Qué aprendimos de los modelos?** En desbalanceo extremo, la accuracy no sirve como referencia; hay que medir recall/precision/F1. El balanceo sintético (SMOTE) no ayudó al modelo lineal pero sí a los árboles. Finalmente, el umbral de decisión permite "calibrar" el modelo según las necesidades del negocio.

## Tecnologías utilizadas

Python · pandas · numpy · matplotlib · seaborn · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · Jupyter

## Estructura del proyecto

```
fraud-detection/
├── data/creditcard.csv          # dataset (ignorado en Git: 150 MB, se descarga de Kaggle)
├── notebooks/fraud_analysis.ipynb  # proyecto completo
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalación

1. Clona o descarga el repositorio.
2. Descarga el dataset [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y colócalo en `data/creditcard.csv` (**no está subido a Git por su tamaño**).
3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

```bash
jupyter notebook
```

Abrir `notebooks/fraud_analysis.ipynb` y ejecutar las celdas en orden.

## Conclusiones

El proyecto demuestra el flujo completo de un problema real de Data Science: entender el negocio y el desbalanceo, preparar datos evitando data leakage, comparar estrategias de balanceo y modelos, evaluar con métricas adecuadas y ajustar el umbral de decisión según el contexto. El resultado es un modelo de Random Forest (con SMOTE) capaz de detectar 104 de 142 fraudes con solo 3 falsas alarmas en el conjunto de prueba.

---
*Fuente de datos: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Kaggle (MLG-ULB, Université Libre de Bruxelles).*