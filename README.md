# 🏦 ¿A quién aprobarle el crédito? – Umbral óptimo de riesgo crediticio

Dashboard que traduce los errores de un modelo de riesgo de crédito a dinero y muestra cuál es el umbral de decisión que minimiza el costo total para el banco.

**Flujo:** 📂 Base > 100.000 clientes → 🧹 Pipeline sin fuga de datos → 🌲 5 modelos × 2 etapas (SMOTE / pesos) → 🏆 Mejor modelo por CV → 💰 Costos FP y FN → 🎯 Umbral óptimo → 📊 Dashboard

🔗 **App publicada:** https://c1-machine-learning-udd.streamlit.app

Certamen 1 · Machine Learning · Universidad del Desarrollo · Docente: Danilo Gómez Correa

---

## 1. Problema de negocio
El banco debe decidir, para cada solicitud, si aprueba o rechaza el crédito. El modelo estima la **probabilidad de impago** y, si supera un umbral, el crédito se rechaza. Los dos errores no cuestan lo mismo:

| Error | Qué ocurre | Costo |
|---|---|---|
| **Falso negativo (FN)** | Se aprueba a un cliente que cae en impago | **US$ 12.150** (75% de un crédito de US$ 16.200 que no se recupera) |
| **Falso positivo (FP)** | Se rechaza a un cliente que sí habría pagado | **US$ 599** (margen de interés perdido en 2 años) |

Como un FN cuesta **20,3 veces** más que un FP, el umbral por defecto de 0,5 no es el adecuado. Los supuestos de costo siguen a Correa Bahnsen et al. (2014), que trabajan sobre esta misma base.

## 2. Datos
| Campo | Detalle |
|---|---|
| Dataset | *Give Me Some Credit* (archivo `cs-training.csv`) |
| Fuente | Kaggle: https://www.kaggle.com/c/GiveMeSomeCredit/data |
| Registros | 150.000 clientes → **149.391** tras eliminar 609 duplicados |
| Variable objetivo | `SeriousDlqin2yrs`: 1 = atraso de 90 días o más en los 2 años siguientes |
| Clase minoritaria | **6,7%** de impago (10.009 clientes) |
| Predictoras | 10 numéricas: edad, dependientes, ingreso, `DebtRatio`, utilización de crédito, líneas abiertas, hipotecas y 3 historiales de atraso |
| Partición | Train / test **80/20 estratificada**, semilla 42 (119.512 / 29.879 clientes) |

**Limpieza (dentro del pipeline, con `ColumnTransformer`):**
- Los códigos 96/98 de las columnas de atraso pasan a nulo y se agrega un indicador 0/1. Esos 225 clientes tienen un 60,4% de impago.
- Imputación por mediana con indicador de nulo para `MonthlyIncome` (19,6% de nulos) y `NumberOfDependents`.
- Recorte al percentil 99 de los montos y conteos. Las edades menores a 18 pasan a nulo.
- Las medianas y los percentiles se calculan **solo con el train de cada fold**, para que no haya fuga de datos.

El dataset no se sube a este repositorio. Se descarga desde el enlace de Kaggle.

## 3. Modelos y resultados

### Parte 1 – Comparación en dos etapas (10 combinaciones)
Se compararon cinco modelos: árbol de decisión, Random Forest, XGBoost, AdaBoost y LightGBM. Cada uno se probó en dos etapas:
- **Etapa A:** SMOTE como paso del `Pipeline` de imblearn.
- **Etapa B:** pesos ponderados (`class_weight="balanced"` o `scale_pos_weight`).

Los hiperparámetros se buscaron con `RandomizedSearchCV` (10 combinaciones, `StratifiedKFold` de 3 folds) sobre una submuestra estratificada de 30.000 clientes. La métrica fue el **F1 de la clase impago**.

| Etapa | Modelo | F1 CV | F1 test | Brecha train–test | Recall | Precisión | AUC |
|---|---|---|---|---|---|---|---|
| **A – SMOTE** | **Random Forest** | **0,428 ± 0,011** | **0,448** | 0,005 | 0,488 | 0,414 | 0,858 |
| A – SMOTE | AdaBoost | 0,420 ± 0,012 | 0,427 | −0,007 | 0,456 | 0,402 | 0,844 |
| A – SMOTE | XGBoost | 0,420 ± 0,014 | 0,423 | −0,004 | 0,501 | 0,365 | 0,847 |
| A – SMOTE | LightGBM | 0,417 ± 0,012 | 0,422 | −0,004 | 0,433 | 0,412 | 0,848 |
| A – SMOTE | Árbol | 0,394 ± 0,020 | 0,403 | 0,000 | 0,479 | 0,348 | 0,823 |
| B – Pesos | LightGBM | 0,389 ± 0,014 | 0,349 | 0,043 | 0,751 | 0,227 | 0,862 |
| B – Pesos | AdaBoost | 0,367 ± 0,014 | 0,369 | −0,001 | 0,337 | 0,408 | 0,782 |
| B – Pesos | XGBoost | 0,364 ± 0,011 | 0,345 | 0,053 | 0,734 | 0,225 | 0,854 |
| B – Pesos | Random Forest | 0,349 ± 0,012 | 0,356 | 0,024 | 0,760 | 0,232 | 0,863 |
| B – Pesos | Árbol | 0,313 ± 0,008 | 0,310 | 0,005 | 0,807 | 0,192 | 0,852 |

**Mejor modelo: Random Forest con SMOTE** (`n_estimators=100`, `max_depth=15`, `min_samples_leaf=50`, `max_features="sqrt"`). Se eligió por el F1 de validación cruzada; el test solo confirma. Tiene el mejor F1 CV, la menor desviación entre folds de la Etapa A y una brecha de solo 0,005. Los modelos con pesos tienen un F1 menor con umbral 0,5, pero un AUC igual o mayor: ordenan igual de bien a los clientes, y la diferencia está en el umbral.

### Parte 2 – Costo y umbral óptimo
1. **Probabilidades en escala real.** El modelo se reentrenó sin SMOTE ni pesos, con los mismos hiperparámetros. Con SMOTE la probabilidad media de impago era 0,169; en escala real es 0,068, igual a la tasa real de 6,7%.
2. **Umbral teórico:** p\* = C_FP / (C_FP + C_FN) = **0,047**
3. **Umbral empírico:** `TunedThresholdClassifierCV`, usando solo el train, elige **0,055**.

| Estrategia (test, 29.879 clientes) | Umbral | Recall | Precisión | Costo total | Ahorro vs. 0,5 |
|---|---|---|---|---|---|
| Aprobar a todos | – | 0,0% | – | US$ 24.324.300 | −13,8% |
| Random Forest, umbral 0,5 | 0,500 | 12,5% | 61,1% | US$ 21.370.554 | – |
| Rechazar a todos | – | 100% | 6,7% | US$ 16.709.474 | 21,8% |
| Random Forest con SMOTE (Etapa A) | 0,500 | 48,8% | 41,4% | US$ 13.292.473 | 37,8% |
| Random Forest con pesos (Etapa B) | 0,500 | 76,0% | 23,2% | US$ 8.860.331 | 58,5% |
| Random Forest, umbral teórico p\* | 0,047 | 86,1% | 16,3% | US$ 8.675.797 | 59,4% |
| **Random Forest, umbral CV** | **0,055** | **84,4%** | **17,8%** | **US$ 8.468.680** | **60,4%** |

**Sensibilidad.** Se probaron seis escenarios de la razón FN/FP: 1, 2, 4,5, 10, 20,3 y 30. En todos los escenarios con FN más caro que FP, bajar el umbral supera a 0,5. El ahorro va de 9,8% (r = 2) a 67,5% (r = 30). SMOTE aplica siempre una razón implícita de ~13,9 y no se adapta a los costos del negocio.

**Recomendación:** usar el Random Forest en escala real con umbral **0,055**, es decir, rechazar las solicitudes con una probabilidad de impago de 5,5% o más. Ahorra **US$ 12,9 millones (60,4%)** frente al umbral 0,5.

## 4. App
Hecha con **Streamlit** y publicada en **Streamlit Community Cloud**. Usa las probabilidades del test del modelo en escala real (`dashboard_test.csv`) y los parámetros base (`parametros.json`), ambos exportados desde el notebook. Incluye:

1. **Controles de costos:** el costo de un FP y de un FN se pueden editar, y hay escenarios predefinidos (caso base, Basilea II, pérdida alta y costos iguales). El umbral p\* se recalcula en vivo.
2. **Umbral manual:** un slider para mover el umbral y compararlo con el óptimo, más un botón para saltar al umbral de costo mínimo.
3. **Curva de costo total vs. umbral:** marca el umbral 0,5, p\*, el mínimo y el umbral elegido.
4. **Matriz de confusión** del umbral elegido, con el costo en dinero de cada error.
5. **Indicadores:** recall, precisión, costo total, % de clientes rechazados y ahorro frente al umbral 0,5.
6. **Opcional:** curva ROC con la recta de iso-costo.

El CSV contiene solo la clase real y la probabilidad estimada de cada cliente del test. No incluye variables personales.

## 5. Estructura
```
├── app.py               # dashboard Streamlit
├── requirements.txt     # librerías
├── dashboard_test.csv   # y_test y probabilidad de impago (29.879 clientes del test)
├── parametros.json      # costos base, umbral teórico y umbral CV
└── README.md
```

## 6. Cómo ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 7. Integrantes
- Eduardo Avendaño
- Javiera García
- Fernando Rodríguez

## 8. Declaración de uso de IA
- **Claude (Anthropic):** apoyo para generar y explicar código del notebook (pipeline, búsqueda de hiperparámetros, umbral y sensibilidad), construir `app.py` y redactar este README. El grupo adaptó el código a la base y al problema, ejecutó los modelos, interpretó los resultados y revisó cada parte.

## 9. Consideraciones éticas
- **Sesgo por edad:** con el umbral recomendado, se rechaza al 49,9% de los buenos clientes menores de 30 años, frente al 8,1% de los mayores de 75. Una parte refleja un riesgo real (11,8% vs. 2,1% de impago), pero el umbral bajo amplifica la diferencia. Antes de usar el modelo, el banco debería revisar la normativa sobre el uso de la edad y monitorear los rechazos por tramo.
- **Privacidad:** la base es anónima, sin nombres, RUT ni datos de contacto.
- **Supuestos:** los costos de FP y FN son supuestos declarados. El costo del FP considera solo el margen perdido y podría estar subestimado.
- Prototipo académico. No reemplaza la evaluación de crédito de un analista ni la normativa vigente.
