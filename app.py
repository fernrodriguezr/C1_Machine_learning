"""
Dashboard — Certamen 1 Machine Learning (UDD)
Costo de clasificación y umbral óptimo para riesgo de crédito (Give Me Some Credit).

Lee dos archivos exportados desde el notebook (sección 9):
  - dashboard_test.csv : y_test (clase real) y proba (probabilidad de impago del modelo en escala real)
  - parametros.json    : costos base (c_fp, c_fn), umbral teórico y umbral por validación cruzada
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Umbral óptimo de crédito", page_icon="🏦", layout="wide")

# ----------------------------------------------------------------------------
# Colores (estética bancaria) y formato de números
# ----------------------------------------------------------------------------
AZUL = "#1f3b63"
AZUL_CLARO = "#4f7cac"
NARANJO = "#e0812f"
VERDE = "#2e8b57"
ROJO = "#c0392b"
GRIS = "#8a8f98"


def usd(x):
    """Formato US$ con punto de miles (estilo chileno)."""
    signo = "-" if x < 0 else ""
    return f"{signo}US$ {abs(x):,.0f}".replace(",", ".")


def usd_m(x):
    """Formato corto en millones para los indicadores."""
    return f"US$ {x / 1e6:.2f} M".replace(".", ",")


def pct(x):
    return f"{x * 100:.1f}%".replace(".", ",")


def dec(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    datos = pd.read_csv("dashboard_test.csv")
    with open("parametros.json", encoding="utf-8") as f:
        parametros = json.load(f)
    return datos["y_test"].to_numpy(), datos["proba"].to_numpy(), parametros


y, proba, par = cargar_datos()
P = int(y.sum())          # impagos reales en test
N = int(len(y) - P)       # buenos clientes en test
UMBRALES = np.round(np.arange(0.005, 1.0, 0.005), 3)


@st.cache_data
def conteos_por_umbral(_y, _proba, umbrales):
    """TP, FP, FN, TN para cada umbral (no depende de los costos, se calcula una vez)."""
    pred = _proba[None, :] >= umbrales[:, None]
    tp = (pred & (_y == 1)).sum(axis=1)
    fp = (pred & (_y == 0)).sum(axis=1)
    return tp, fp, P - tp, N - fp


def matriz(umbral):
    pred = proba >= umbral
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return tp, fp, P - tp, N - fp


TP_G, FP_G, FN_G, TN_G = conteos_por_umbral(y, proba, UMBRALES)

# ----------------------------------------------------------------------------
# Escenarios de costos (para la demo en vivo)
# ----------------------------------------------------------------------------
C_FP_BASE = float(par["c_fp"])
C_FN_BASE = float(par["c_fn"])
ESCENARIOS = {
    f"Caso base (FN/FP = {C_FN_BASE / C_FP_BASE:.1f})".replace(".", ","): (C_FP_BASE, C_FN_BASE),
    "Basilea II (FN/FP = 4,5)": (C_FP_BASE, C_FP_BASE * 4.5),
    "Pérdida alta (FN/FP = 30)": (C_FP_BASE, C_FP_BASE * 30),
    "Costos iguales (FN/FP = 1)": (C_FP_BASE, C_FP_BASE),
}

if "c_fp" not in st.session_state:
    st.session_state.c_fp = round(C_FP_BASE, 2)
    st.session_state.c_fn = round(C_FN_BASE, 2)
    st.session_state.umbral = float(par["umbral_cv"])


def aplicar_escenario():
    c_fp, c_fn = ESCENARIOS[st.session_state.escenario]
    st.session_state.c_fp = round(c_fp, 2)
    st.session_state.c_fn = round(c_fn, 2)


def ir_al_optimo():
    st.session_state.umbral = st.session_state.umbral_optimo


# ----------------------------------------------------------------------------
# Barra lateral: controles
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Costos de cada error")
    st.selectbox("Escenario", list(ESCENARIOS), key="escenario", on_change=aplicar_escenario)
    st.number_input("Costo de un falso positivo (US$)", min_value=1.0, step=50.0, key="c_fp",
                    help="Rechazar a un cliente que sí habría pagado: margen de interés perdido.")
    st.number_input("Costo de un falso negativo (US$)", min_value=1.0, step=500.0, key="c_fn",
                    help="Aprobar a un cliente que cae en impago: parte del crédito que no se recupera.")

    c_fp, c_fn = st.session_state.c_fp, st.session_state.c_fn
    p_star = c_fp / (c_fp + c_fn)

    costos = c_fp * FP_G + c_fn * FN_G
    i_min = int(np.argmin(costos))
    st.session_state.umbral_optimo = float(UMBRALES[i_min])

    st.metric("Umbral teórico p* = C_FP / (C_FP + C_FN)", dec(p_star))
    st.caption(f"Razón FN/FP = {dec(c_fn / c_fp, 1)}. Mientras más caro el FN, más bajo el umbral.")

    st.header("2. Umbral manual")
    st.slider("Umbral de decisión", 0.005, 0.995, step=0.005, key="umbral", format="%.3f",
              help="Se rechaza el crédito si P(impago) ≥ umbral.")
    st.button("Ir al umbral de costo mínimo", on_click=ir_al_optimo, width="stretch")

    st.divider()
    st.caption(f"Modelo: {par['modelo']}. Datos: {len(y):,} clientes del test ".replace(",", ".")
               + f"({pct(P / len(y))} de impago). Umbral por validación cruzada (caso base): "
               + f"{dec(par['umbral_cv'])}.")

umbral = st.session_state.umbral
umbral_opt = st.session_state.umbral_optimo


def costo(tp, fp, fn, tn):
    return c_fp * fp + c_fn * fn


tp, fp, fn, tn = matriz(umbral)
tp5, fp5, fn5, tn5 = matriz(0.5)
tps, fps, fns, tns = matriz(p_star)
costo_sel = costo(tp, fp, fn, tn)
costo_05 = costo(tp5, fp5, fn5, tn5)
costo_pstar = costo(tps, fps, fns, tns)
costo_opt = float(costos[i_min])
ahorro = costo_05 - costo_sel
recall = tp / P if P else 0
precision = tp / (tp + fp) if (tp + fp) else 0

# ----------------------------------------------------------------------------
# Encabezado e indicadores
# ----------------------------------------------------------------------------
st.title("🏦 ¿A quién aprobarle el crédito?")
st.markdown(
    f"El modelo estima la **probabilidad de impago** de cada cliente. Si supera el umbral, se **rechaza** "
    f"el crédito. Con el umbral elegido (**{dec(umbral)}**), así le va al banco en "
    + f"{len(y):,}".replace(",", ".") + " clientes de prueba:")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Costo total", usd_m(costo_sel), help=usd(costo_sel))
k2.metric("Ahorro vs. umbral 0,5", usd_m(ahorro), help=usd(ahorro), delta=f"{ahorro / costo_05 * 100:.1f}%".replace(".", ","), delta_color="normal")
k3.metric("Recall (impagos detectados)", pct(recall), help="De los clientes que caen en impago, % que el modelo rechaza.")
k4.metric("Precisión", pct(precision), help="De los clientes rechazados, % que realmente caería en impago.")
k5.metric("Clientes rechazados", pct((tp + fp) / len(y)))

# ----------------------------------------------------------------------------
# Comparación de umbrales
# ----------------------------------------------------------------------------
st.subheader("Umbral elegido vs. umbral óptimo")
comp = pd.DataFrame([
    ("Umbral 0,5 (por defecto)", 0.5, matriz(0.5)),
    ("Umbral teórico p*", p_star, (tps, fps, fns, tns)),
    ("Umbral de costo mínimo (curva)", umbral_opt, matriz(umbral_opt)),
    ("Umbral elegido (manual)", umbral, (tp, fp, fn, tn)),
], columns=["Estrategia", "Umbral", "m"])
comp["Costo"] = comp["m"].apply(lambda m: costo(*m))
comp["Ahorro vs. 0,5"] = costo_05 - comp["Costo"]
comp["Recall"] = comp["m"].apply(lambda m: m[0] / P)
comp["Precisión"] = comp["m"].apply(lambda m: m[0] / (m[0] + m[1]) if (m[0] + m[1]) else 0)
comp["Impagos aprobados (FN)"] = comp["m"].apply(lambda m: m[2])
comp["Buenos rechazados (FP)"] = comp["m"].apply(lambda m: m[1])
tabla = comp.drop(columns="m").copy()
tabla["Umbral"] = tabla["Umbral"].map(dec)
for c in ["Costo", "Ahorro vs. 0,5"]:
    tabla[c] = tabla[c].map(usd)
for c in ["Recall", "Precisión"]:
    tabla[c] = tabla[c].map(pct)
st.dataframe(tabla, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# Curva de costo vs. umbral y matriz de confusión
# ----------------------------------------------------------------------------
col_curva, col_matriz = st.columns([3, 2])

with col_curva:
    st.subheader("Costo total según el umbral")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=UMBRALES, y=costos / 1e6, mode="lines", name="Costo total",
                             line=dict(color=AZUL, width=3),
                             hovertemplate="Umbral %{x:.3f}<br>Costo US$ %{y:.2f} M<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=GRIS, dash="dot"))
    fig.add_vline(x=p_star, line=dict(color=ROJO, dash="dash"))
    fig.add_vline(x=umbral, line=dict(color=NARANJO, width=2))
    puntos = [(0.5, costo_05, "Umbral 0,5", GRIS), (umbral_opt, costo_opt, "Mínimo", VERDE),
              (umbral, costo_sel, "Elegido", NARANJO)]
    for x, c, nombre, color in puntos:
        fig.add_trace(go.Scatter(x=[x], y=[c / 1e6], mode="markers", name=f"{nombre} ({dec(x)})",
                                 marker=dict(color=color, size=13, line=dict(color="white", width=2)),
                                 hovertemplate=f"{nombre}<br>Umbral %{{x:.3f}}<br>Costo US$ %{{y:.2f}} M<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"p* teórico ({dec(p_star)})",
                             line=dict(color=ROJO, dash="dash")))
    fig.update_layout(xaxis_title="Umbral de decisión", yaxis_title="Costo total (millones de US$)",
                      height=420, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")

with col_matriz:
    st.subheader("Matriz de confusión")
    z = [[tn, fp], [fn, tp]]
    texto = [[f"<b>Aprobado y paga</b><br>{tn:,} clientes<br>US$ 0".replace(",", "."),
              f"<b>Rechazado y pagaría (FP)</b><br>{fp:,} clientes<br>{usd(fp * c_fp)}".replace(",", ".")],
             [f"<b>Aprobado e impaga (FN)</b><br>{fn:,} clientes<br>{usd(fn * c_fn)}".replace(",", "."),
              f"<b>Rechazado e impagaría</b><br>{tp:,} clientes<br>US$ 0".replace(",", ".")]]
    fig_m = go.Figure(go.Heatmap(z=[[0, fp * c_fp], [fn * c_fn, 0]], text=texto, texttemplate="%{text}",
                                 colorscale=[[0, "#eaf2ea"], [1, "#f4c7a1"]], showscale=False,
                                 x=["Aprobar (pred. 0)", "Rechazar (pred. 1)"],
                                 y=["Paga (real 0)", "Impago (real 1)"], hoverinfo="skip",
                                 textfont=dict(size=13, color="#1b1b1b")))
    fig_m.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_m, width="stretch")
    st.caption(f"Costo FP: {usd(fp * c_fp)} · Costo FN: {usd(fn * c_fn)} · **Total: {usd(costo_sel)}**")

# ----------------------------------------------------------------------------
# Opcional: curva ROC con la recta de iso-costo
# ----------------------------------------------------------------------------
with st.expander("Curva ROC con recta de iso-costo (opcional)"):
    tpr = TP_G / P
    fpr = FP_G / N
    pendiente = (c_fp * N) / (c_fn * P)   # todas las combinaciones (FPR, TPR) con el mismo costo
    x0, y0 = fpr[i_min], tpr[i_min]
    xs = np.linspace(0, 1, 50)
    ys = y0 + pendiente * (xs - x0)
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=np.r_[1, fpr, 0], y=np.r_[1, tpr, 0], mode="lines", name="ROC",
                               line=dict(color=AZUL, width=3)))
    fig_r.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                               line=dict(color=GRIS, dash="dot")))
    mask = (ys >= 0) & (ys <= 1)
    fig_r.add_trace(go.Scatter(x=xs[mask], y=ys[mask], mode="lines", name=f"Iso-costo (pendiente {dec(pendiente, 2)})",
                               line=dict(color=ROJO, dash="dash")))
    fig_r.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers", name=f"Costo mínimo ({dec(umbral_opt)})",
                               marker=dict(color=VERDE, size=12)))
    fig_r.add_trace(go.Scatter(x=[fp / N], y=[tp / P], mode="markers", name=f"Elegido ({dec(umbral)})",
                               marker=dict(color=NARANJO, size=12)))
    fig_r.update_layout(xaxis_title="Tasa de falsos positivos (buenos clientes rechazados)",
                        yaxis_title="Tasa de verdaderos positivos (recall)", height=450,
                        template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]))
    st.plotly_chart(fig_r, width="stretch")
    st.caption("La recta de iso-costo une los puntos con el mismo costo total. El umbral óptimo es donde "
               "la recta toca la curva ROC: si el FN es más caro, la recta es más plana y el óptimo "
               "se mueve hacia arriba a la derecha (se rechaza a más clientes).")

st.caption("Certamen 1 · Machine Learning · UDD — Eduardo Avendaño, Javiera García, Fernando Rodríguez. "
           "El umbral de costo mínimo se calcula sobre el test solo para ilustrar; la recomendación del "
           "proyecto usa el umbral elegido por validación cruzada con el train.")
