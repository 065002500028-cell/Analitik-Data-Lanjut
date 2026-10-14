import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Analitik Peramalan Permintaan", layout="wide")

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("Parameter Peramalan")
n = st.sidebar.slider("Periode Moving Average (n)", 2, 6, 3)
st.sidebar.markdown("**Bobot Weighted MA (terbaru ke terlama)**")
w1 = st.sidebar.number_input("Bobot periode terbaru (w1)", 0.0, 1.0, 0.5, 0.05)
w2 = st.sidebar.number_input("Bobot periode ke-2 (w2)", 0.0, 1.0, 0.3, 0.05)
w3 = st.sidebar.number_input("Bobot periode ke-3 (w3)", 0.0, 1.0, 0.2, 0.05)
alpha = st.sidebar.slider("Konstanta Smoothing Alpha (α)", 0.05, 0.95, 0.30, 0.05)

weights = [w1, w2, w3]
if abs(sum(weights) - 1) > 1e-6:
    st.sidebar.error("Jumlah bobot harus sama dengan 1.")
    st.stop()

# ---------------------------------------------------------------- Data
st.title("Aplikasi Analitik Peramalan Permintaan")
st.caption("Simulasi metode time series dan ukuran akurasi (Operations Management, Chapter 3)")

default = pd.DataFrame({
    "Periode": ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"],
    "Permintaan": [120, 130, 125, 140, 150, 145, 160, 170, 165, 180, 185, 195],
})
st.subheader("Data Permintaan Historis")
data = st.data_editor(default, num_rows="dynamic", use_container_width=True, hide_index=True)
data = data.dropna()
A = data["Permintaan"].astype(float).to_numpy()
labels = list(data["Periode"].astype(str))
T = len(A)
start = max(n, 3)  # evaluasi pada periode yang sama untuk semua metode
if T < start + 2:
    st.error(f"Data minimal {start + 2} periode.")
    st.stop()


# ---------------------------------------------------------------- Model
def moving_average(a, k):
    f = np.full(len(a) + 1, np.nan)
    for t in range(k, len(a) + 1):
        f[t] = a[t - k:t].mean()
    return f


def weighted_ma(a, w):
    k = len(w)
    f = np.full(len(a) + 1, np.nan)
    for t in range(k, len(a) + 1):
        f[t] = np.dot(a[t - k:t][::-1], w)
    return f


def exp_smoothing(a, al):
    f = np.full(len(a) + 1, np.nan)
    f[0] = a[0]
    for t in range(1, len(a) + 1):
        f[t] = f[t - 1] + al * (a[t - 1] - f[t - 1])
    return f


def accuracy(a, f, s):
    x, y = a[s:], f[s:len(a)]
    e = x - y
    return {"MAD": np.mean(np.abs(e)), "MSE": np.mean(e ** 2), "MAPE (%)": np.mean(np.abs(e) / x) * 100}


models = {
    f"Moving Average (n={n})": moving_average(A, n),
    "Weighted Moving Average": weighted_ma(A, weights),
    f"Exponential Smoothing (α={alpha:.2f})": exp_smoothing(A, alpha),
}

# ---------------------------------------------------------------- Hasil
st.subheader("Hasil Peramalan")
table = pd.DataFrame({"Periode": labels + ["Periode Berikutnya"], "Aktual": list(A) + [np.nan]})
for name, f in models.items():
    table[name] = f
st.dataframe(table.style.format("{:,.2f}", subset=table.columns[1:], na_rep="-"),
             hide_index=True, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=labels, y=A, name="Aktual", mode="lines+markers", line=dict(width=3)))
for name, f in models.items():
    fig.add_trace(go.Scatter(x=labels + ["Berikutnya"], y=f, name=name, mode="lines+markers"))
fig.update_layout(title="Permintaan Aktual vs Peramalan", xaxis_title="Periode", yaxis_title="Unit")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Ukuran Akurasi Peramalan")
st.caption(f"Dihitung pada periode yang sama untuk semua metode (periode ke-{start + 1} sampai ke-{T}).")
acc = pd.DataFrame({name: accuracy(A, f, start) for name, f in models.items()}).T
acc["Peramalan Berikutnya"] = [f[-1] for f in models.values()]
st.dataframe(acc.style.format("{:,.2f}"), use_container_width=True)

best = acc["MAD"].idxmin()
st.success(f"**Rekomendasi:** {best} memiliki MAD terendah ({acc.loc[best, 'MAD']:,.2f} unit), "
           f"dengan peramalan periode berikutnya {acc.loc[best, 'Peramalan Berikutnya']:,.2f} unit.")

# ---------------------------------------------------------------- Rincian
with st.expander("Rumus yang digunakan"):
    st.latex(r"F_{t+1}=\frac{A_t+A_{t-1}+\dots+A_{t-n+1}}{n}")
    st.latex(r"F_{t+1}=w_1A_t+w_2A_{t-1}+w_3A_{t-2},\quad \sum w=1")
    st.latex(r"F_{t+1}=F_t+\alpha\,(A_t-F_t)")
    st.latex(r"MAD=\frac{\sum|e_t|}{n},\quad MSE=\frac{\sum e_t^2}{n},\quad MAPE=\frac{100}{n}\sum\frac{|e_t|}{A_t}")
