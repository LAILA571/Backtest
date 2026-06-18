"""
=====================================================================
APPLICATION STREAMLIT — BACK-TESTING DE LA VaR
Traffic Light (Bâle) · Kupiec (POF) · Christoffersen (indépendance &
couverture conditionnelle) · Exigence en fonds propres
=====================================================================

Conçue pour le fichier de type "Classeur2.xlsx" avec les colonnes :
    value_date | hypothetical_value | previous_value |
    pnl_hypothétique_t | VAR_t-1

Règle de violation :  I_t = 1  si  pnl_hypothétique_t < VAR_{t-1}  sinon 0

---------------------------------------------------------------------
LANCEMENT
---------------------------------------------------------------------
    pip install streamlit pandas numpy scipy plotly openpyxl
    streamlit run app_backtesting_complet.py
---------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(page_title="Back-testing VaR — Analyse complète",
                   layout="wide")

CHI2_1 = 3.841   # chi-deux 95 %, 1 ddl
CHI2_2 = 5.991   # chi-deux 95 %, 2 ddl

# Tableau réglementaire du multiplicateur de Bâle
MULTIPLIER = {0: 3.00, 1: 3.00, 2: 3.00, 3: 3.00, 4: 3.00,
              5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}


# =====================================================================
#  FONCTIONS DE CALCUL
# =====================================================================
def log_term(n, k, pr):
    """k*ln(pr) + (n-k)*ln(1-pr), avec gestion des cas limites."""
    a = (n - k) * np.log(1 - pr) if (n - k) > 0 and (1 - pr) > 0 else 0.0
    b = k * np.log(pr) if (k > 0 and pr > 0) else 0.0
    return a + b


def kupiec_pof(T, N, p):
    phat = N / T if T else 0.0
    lr = 2 * (log_term(T, N, phat) - log_term(T, N, p))
    return lr, stats.chi2.sf(lr, 1), phat


def transition_counts(I):
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(I)):
        a, b = I[i - 1], I[i]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    return n00, n01, n10, n11


def christoffersen_ind(I):
    n00, n01, n10, n11 = transition_counts(I)
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    tot = n00 + n01 + n10 + n11
    pi = (n01 + n11) / tot if tot else 0.0

    def tt(c, pr):
        return c * np.log(pr) if (c > 0 and pr > 0) else 0.0

    ll_dep = (tt(n00, 1 - pi0) + tt(n01, pi0)
              + tt(n10, 1 - pi1) + tt(n11, pi1))
    ll_ind = tt(n00 + n10, 1 - pi) + tt(n01 + n11, pi)
    lr = 2 * (ll_dep - ll_ind)
    return lr, stats.chi2.sf(lr, 1), (n00, n01, n10, n11, pi0, pi1, pi)


def basel_zone(N):
    if N <= 4:
        return "VERTE", "#2e7d32"
    if N <= 9:
        return "JAUNE", "#f9a825"
    return "ROUGE", "#c62828"


def fmt(v, dec=4):
    return f"{v:.{dec}f}"


def money(v):
    return f"{v:,.0f}".replace(",", " ")


# =====================================================================
#  LECTURE DES DONNÉES
# =====================================================================
@st.cache_data
def load(file):
    name = file.name.lower()
    df = (pd.read_excel(file) if name.endswith((".xlsx", ".xls"))
          else pd.read_csv(file, sep=None, engine="python"))
    # normalisation des noms de colonnes
    ren = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "value_date" or "date" in cl:
            ren[c] = "value_date"
        elif "pnl" in cl:
            ren[c] = "pnl"
        elif cl.startswith("var") or "var_t" in cl or cl == "var":
            ren[c] = "var"
        elif "hypothetical" in cl:
            ren[c] = "hypothetical_value"
        elif "previous" in cl:
            ren[c] = "previous_value"
    df = df.rename(columns=ren)
    if "pnl" not in df.columns and {"hypothetical_value", "previous_value"} <= set(df.columns):
        df["pnl"] = df["hypothetical_value"] - df["previous_value"]
    if not {"value_date", "pnl", "var"} <= set(df.columns):
        raise ValueError("Colonnes attendues : value_date, pnl_hypothétique_t, VAR_t-1.")
    df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce")
    for c in ["pnl", "var"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["value_date", "pnl", "var"]).sort_values("value_date")
    return df.reset_index(drop=True)


# =====================================================================
#  INTERFACE
# =====================================================================
st.title("Back-testing de la Value-at-Risk — Analyse réglementaire complète")
st.caption("Traffic Light de Bâle · Test de Kupiec · Tests de Christoffersen · "
           "Exigence en fonds propres")

with st.sidebar:
    st.header("Paramètres")
    up = st.file_uploader("Fichier de données (Excel ou CSV)",
                          type=["xlsx", "xls", "csv"])
    alpha = st.slider("Niveau de confiance de la VaR (α)",
                      0.90, 0.995, 0.99, 0.005, format="%.3f")
    st.markdown("---")
    st.caption("Violation : **I_t = 1 si pnl_hypo_t < VaR_{t−1}**, sinon 0.")

if up is None:
    st.info("⬅️ Charge le fichier **Classeur2.xlsx** (colonnes value_date, "
            "pnl_hypothétique_t, VAR_t-1) pour lancer l'analyse.")
    st.stop()

try:
    df = load(up)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---- calculs principaux ----
p = 1 - alpha
df["violation"] = (df["pnl"] < df["var"]).astype(int)
I = df["violation"].values
T = len(I)
N = int(I.sum())
phat = N / T
expected = T * p
zone, zcolor = basel_zone(N)

# =====================================================================
#  1. PRÉSENTATION DES DONNÉES
# =====================================================================
st.header("1. Présentation des données et indicatrice de violation")
st.markdown(
    "Pour chaque date *t*, on compare le **P&L hypothétique** au seuil de **VaR "
    "de la veille**. L'indicatrice de violation vaut :")
st.latex(r"I_t = \begin{cases} 1 & \text{si } \text{pnl}_t < \text{VaR}_{t-1} "
         r"\quad(\text{dépassement})\\[2pt] 0 & \text{sinon} \end{cases}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations T", T)
c2.metric("Violations N", N)
c3.metric("Violations attendues  T·p", f"{expected:.2f}")
c4.metric("Fréquence observée  p̂ = N/T", f"{phat:.2%}")

show = df.copy()
show["value_date"] = show["value_date"].dt.strftime("%d/%m/%Y")
show = show.rename(columns={"value_date": "Date", "pnl": "pnl_hypo_t",
                            "var": "VaR_t-1", "violation": "I_t"})
cols = [c for c in ["Date", "hypothetical_value", "previous_value",
                    "pnl_hypo_t", "VaR_t-1", "I_t"] if c in show.columns]
st.dataframe(show[cols], use_container_width=True, height=300, hide_index=True)

# graphique
st.markdown("**Évolution du P&L hypothétique encadré par ± VaR** "
            "(les dépassements sont marqués en rouge) :")
fig = go.Figure()
colors = np.where(I == 1, "#c62828", "#4472C4")
fig.add_bar(x=df["value_date"], y=df["pnl"], name="P&L hypothétique",
            marker_color=colors)
fig.add_scatter(x=df["value_date"], y=df["var"], name="VaR (−)",
                mode="lines", line=dict(color="#c62828", width=1.6))
fig.add_scatter(x=df["value_date"], y=-df["var"], name="+ VaR",
                mode="lines", line=dict(color="#2e7d32", width=1.6))
vio = df[df["violation"] == 1]
fig.add_scatter(x=vio["value_date"], y=vio["pnl"], name="Violation",
                mode="markers", marker=dict(color="#c62828", size=11, symbol="x"))
fig.update_layout(height=430, hovermode="x unified",
                  legend=dict(orientation="h", y=1.08),
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# =====================================================================
#  2. DATES DES VIOLATIONS
# =====================================================================
st.header("2. Dates des violations")
if N == 0:
    st.success("Aucune violation observée sur la période.")
else:
    vtab = vio.copy()
    vtab["value_date"] = vtab["value_date"].dt.strftime("%d/%m/%Y")
    vtab["écart (pnl − VaR)"] = (vtab["pnl"] - vtab["var"])
    vtab = vtab[["value_date", "pnl", "var", "écart (pnl − VaR)"]].rename(
        columns={"value_date": "Date du dépassement", "pnl": "pnl_hypo_t",
                 "var": "VaR_t-1"})
    st.dataframe(vtab, use_container_width=True, hide_index=True)
    st.caption("Un écart négatif confirme que la perte a dépassé la VaR ce jour-là.")

# =====================================================================
#  3. TRAFFIC LIGHT (BÂLE)
# =====================================================================
st.header("3. Approche réglementaire de Bâle — Traffic Light Test")
st.markdown(
    "Le test du feu tricolore classe le modèle selon le **nombre total de "
    "violations**. Le nombre attendu sur *T* jours est :")
st.latex(r"E[N] = T \times (1-\alpha) = %d \times %.4f = %.2f" % (T, p, expected))

zt = pd.DataFrame({
    "Zone": ["Verte", "Jaune", "Rouge"],
    "Nombre de violations": ["0 à 4", "5 à 9", "10 et plus"],
    "Interprétation": ["Modèle fiable — aucune pénalité",
                       "Surveillance accrue — majoration du capital",
                       "Modèle déficient — à revoir"],
})
st.table(zt)

cZ1, cZ2 = st.columns([1, 2])
cZ1.markdown(
    f"<div style='padding:18px;border-radius:10px;background:{zcolor}22;"
    f"border:2px solid {zcolor};text-align:center'>"
    f"<div style='font-size:.9rem;color:#555'>Zone du modèle</div>"
    f"<div style='font-size:2rem;font-weight:800;color:{zcolor}'>{zone}</div>"
    f"<div style='color:#555'>{N} violation(s)</div></div>",
    unsafe_allow_html=True)
with cZ2:
    if zone == "VERTE":
        st.success("Le nombre de violations est statistiquement compatible avec "
                   "le modèle. Celui-ci est jugé **fiable** et conserve le "
                   "multiplicateur minimal de 3.")
    elif zone == "JAUNE":
        st.warning("Le nombre de violations est plus élevé qu'attendu sans prouver "
                   "que le modèle est erroné. Le modèle reste **toléré** mais "
                   "fait l'objet d'une surveillance accrue et d'une majoration "
                   "du capital.")
    else:
        st.error("Le nombre de violations est trop important pour être imputable "
                 "au hasard. Le modèle est considéré comme **déficient** et doit "
                 "être révisé (multiplicateur porté à 4).")

# =====================================================================
#  4. TEST DE KUPIEC (POF)
# =====================================================================
st.header("4. Test de Kupiec — Proportion of Failures (couverture non conditionnelle)")
lr_pof, p_pof, _ = kupiec_pof(T, N, p)

st.markdown("**Hypothèses** — H₀ : p̂ = p (fréquence correcte) ; "
            "H₁ : p̂ ≠ p (test bilatéral).")
st.markdown("**Statistique du rapport de vraisemblance :**")
st.latex(r"LR_{POF} = -2\ln\!\left[\frac{(1-p)^{T-N}\,p^{N}}"
         r"{\left(1-\frac{N}{T}\right)^{T-N}\left(\frac{N}{T}\right)^{N}}\right]"
         r"\;\sim\;\chi^2_1")

vk = pd.DataFrame({
    "Variable": ["T (observations)", "N (violations)", "p = 1 − α",
                 "p̂ = N / T", "LR_POF", "Valeur critique χ²(0,95 ; 1)",
                 "p-value"],
    "Valeur": [str(T), str(N), fmt(p, 4), fmt(phat, 4),
               fmt(lr_pof, 4), fmt(CHI2_1, 3), fmt(p_pof, 4)],
})
st.table(vk)

dec_pof = lr_pof > CHI2_1
st.markdown(
    f"**Comparaison au χ² :** LR_POF = **{lr_pof:.4f}** "
    f"{'>' if dec_pof else '≤'} χ²₀,₉₅;₁ = **{CHI2_1}** "
    f"(p-value = {p_pof:.4f}).")
if dec_pof:
    st.error("On **rejette H₀** : la fréquence des violations n'est pas conforme "
             "au niveau de confiance. Le modèle est mal calibré "
             + ("(sous-estimation du risque)." if phat > p
                else "(surestimation du risque)."))
else:
    st.success("On **ne rejette pas H₀** : la fréquence observée des violations "
               "est statistiquement compatible avec le niveau de confiance. "
               "Le modèle est correctement calibré sur le plan de la fréquence.")

# =====================================================================
#  5. TEST D'INDÉPENDANCE DE CHRISTOFFERSEN
# =====================================================================
st.header("5. Test d'indépendance de Christoffersen")
lr_ind, p_ind, (n00, n01, n10, n11, pi0, pi1, pi) = christoffersen_ind(I)

st.markdown("On construit la **matrice de transition** des états de violation "
            "(0 = pas de violation, 1 = violation) :")
mt = pd.DataFrame(
    [[n00, n01, n00 + n01], [n10, n11, n10 + n11],
     [n00 + n10, n01 + n11, n00 + n01 + n10 + n11]],
    index=["t−1 = 0", "t−1 = 1", "Total"],
    columns=["t = 0", "t = 1", "Total"])
st.table(mt)

st.markdown("**Probabilités conditionnelles :**")
colA, colB, colC = st.columns(3)
colA.latex(r"\pi_0 = \frac{n_{01}}{n_{00}+n_{01}} = %.4f" % pi0)
colB.latex(r"\pi_1 = \frac{n_{11}}{n_{10}+n_{11}} = %.4f" % pi1)
colC.latex(r"\pi = \frac{n_{01}+n_{11}}{T} = %.4f" % pi)

st.markdown("**Hypothèses** — H₀ : π₀ = π₁ (indépendance) ; "
            "H₁ : π₀ ≠ π₁ (dépendance / regroupement).")
st.markdown("**Statistique du test :**")
st.latex(r"LR_{ind} = -2\ln\!\left[\frac{(1-\pi)^{n_{00}+n_{10}}\,"
         r"\pi^{\,n_{01}+n_{11}}}{(1-\pi_0)^{n_{00}}\pi_0^{\,n_{01}}"
         r"(1-\pi_1)^{n_{10}}\pi_1^{\,n_{11}}}\right]\;\sim\;\chi^2_1")

vi = pd.DataFrame({
    "Variable": ["n00", "n01", "n10", "n11", "π0", "π1", "π",
                 "LR_ind", "Valeur critique χ²(0,95 ; 1)", "p-value"],
    "Valeur": [str(n00), str(n01), str(n10), str(n11),
               fmt(pi0), fmt(pi1), fmt(pi), fmt(lr_ind, 4),
               fmt(CHI2_1, 3), fmt(p_ind, 4)],
})
st.table(vi)

dec_ind = lr_ind > CHI2_1
st.markdown(
    f"**Comparaison au χ² :** LR_ind = **{lr_ind:.4f}** "
    f"{'>' if dec_ind else '≤'} χ²₀,₉₅;₁ = **{CHI2_1}** "
    f"(p-value = {p_ind:.4f}).")
if dec_ind:
    st.error("On **rejette H₀** : les violations sont **dépendantes** "
             "(regroupement / clustering). Le modèle ne capture pas bien la "
             "dynamique de la volatilité.")
else:
    st.success("On **ne rejette pas H₀** : les violations sont **indépendantes** "
               "et se répartissent aléatoirement dans le temps.")

# =====================================================================
#  6. COUVERTURE CONDITIONNELLE DE CHRISTOFFERSEN
# =====================================================================
st.header("6. Test de couverture conditionnelle de Christoffersen")
lr_cc = lr_pof + lr_ind
p_cc = stats.chi2.sf(lr_cc, 2)
st.latex(r"LR_{cc} = LR_{POF} + LR_{ind} = %.4f + %.4f = %.4f \;\sim\; \chi^2_2"
         % (lr_pof, lr_ind, lr_cc))

vc = pd.DataFrame({
    "Variable": ["LR_POF", "LR_ind", "LR_cc", "Valeur critique χ²(0,95 ; 2)",
                 "p-value"],
    "Valeur": [fmt(lr_pof, 4), fmt(lr_ind, 4), fmt(lr_cc, 4),
               fmt(CHI2_2, 3), fmt(p_cc, 4)],
})
st.table(vc)

dec_cc = lr_cc > CHI2_2
st.markdown(
    f"**Comparaison au χ² :** LR_cc = **{lr_cc:.4f}** "
    f"{'>' if dec_cc else '≤'} χ²₀,₉₅;₂ = **{CHI2_2}** "
    f"(p-value = {p_cc:.4f}).")
if dec_cc:
    st.error("On **rejette H₀** : au moins l'une des deux propriétés (fréquence "
             "ou indépendance) n'est pas respectée — le modèle de VaR est rejeté.")
else:
    st.success("On **ne rejette pas H₀** : le modèle fournit à la fois le bon "
               "nombre de violations et des violations indépendantes — le modèle "
               "de VaR est **validé**.")

# diagnostic combiné
st.markdown("**Diagnostic combiné :**")
if not dec_pof and not dec_ind:
    diag = "Aucun rejet : modèle correct en fréquence et en indépendance."
elif dec_pof and not dec_ind:
    diag = "Rejet sur la fréquence uniquement : mauvaise calibration du nombre de violations."
elif not dec_pof and dec_ind:
    diag = "Rejet sur l'indépendance uniquement : regroupement temporel des violations."
else:
    diag = "Rejet global : défaillance à la fois en calibration et en indépendance."
st.info(diag)

# =====================================================================
#  7. EXIGENCE EN FONDS PROPRES
# =====================================================================
st.header("7. Exigence en fonds propres (Bâle)")
k = MULTIPLIER.get(N, 4.0)
complement = k - 3.0

st.markdown("Le multiplicateur réglementaire **k = 3 + complément** dépend du "
            "nombre de violations :")
mtab = pd.DataFrame({
    "Zone": ["Verte", "Jaune", "Jaune", "Jaune", "Jaune", "Jaune", "Rouge"],
    "Violations": ["0 – 4", "5", "6", "7", "8", "9", "≥ 10"],
    "Facteur k": ["3,00", "3,40", "3,50", "3,65", "3,75", "3,85", "4,00"],
})
st.table(mtab)

st.latex(r"FP_t = \max\!\left(\,VaR_t,\;(3+\text{complément})\times"
         r"\frac{1}{60}\sum_{i=t-60}^{t-1} VaR_i\right)")

av = df["var"].abs()
avg60 = av.iloc[-60:].mean() if len(av) >= 1 else np.nan
var_t = av.iloc[-1]
term2 = k * avg60
FP = max(var_t, term2)

vf = pd.DataFrame({
    "Élément": ["Nombre de violations N", "Multiplicateur k = 3 + complément",
                "|VaR_t| (dernière date)",
                "Moyenne des |VaR| sur 60 jours",
                "k × moyenne₆₀", "Exigence FP_t = max(·)"],
    "Valeur": [str(N), f"{k:.2f}  (complément = {complement:.2f})",
               money(var_t), money(avg60), money(term2), money(FP)],
})
st.table(vf)

cF1, cF2 = st.columns([1, 2])
cF1.metric("Exigence en fonds propres (FP_t)", money(FP))
cF2.info("L'exigence retient le maximum entre la VaR du jour et le produit du "
         f"multiplicateur (**k = {k:.2f}**) par la moyenne des VaR des 60 derniers "
         "jours. Un modèle en zone verte conserve k = 3 (capital minimal) ; un "
         "modèle dégradé subit un k plus élevé, donc une exigence accrue.")

# =====================================================================
#  8. SYNTHÈSE POUR LE RAPPORT
# =====================================================================
st.header("8. Synthèse")
synth = pd.DataFrame({
    "Test": ["Traffic Light (Bâle)", "Kupiec POF (df=1)",
             "Christoffersen indépendance (df=1)",
             "Christoffersen couv. cond. (df=2)"],
    "Statistique / Indicateur": [f"{N} violations",
                                 f"LR = {lr_pof:.4f}",
                                 f"LR = {lr_ind:.4f}",
                                 f"LR = {lr_cc:.4f}"],
    "Seuil": ["zone", str(CHI2_1), str(CHI2_1), str(CHI2_2)],
    "p-value": ["—", f"{p_pof:.4f}", f"{p_ind:.4f}", f"{p_cc:.4f}"],
    "Décision": [f"Zone {zone.lower()}",
                 "Rejet" if dec_pof else "Non rejeté",
                 "Rejet" if dec_ind else "Non rejeté",
                 "Rejet" if dec_cc else "Non rejeté"],
})
st.dataframe(synth, use_container_width=True, hide_index=True)

valide = (not dec_cc) and (zone != "ROUGE")
if valide:
    st.success(f"**Conclusion.** Sur {T} jours, le modèle enregistre {N} violations "
               f"pour {expected:.2f} attendues. Tous les tests statistiques "
               f"valident le modèle (zone {zone.lower()}), qui est donc jugé "
               f"**fiable**. L'exigence en fonds propres s'élève à {money(FP)} "
               f"avec le multiplicateur minimal k = {k:.2f}.")
else:
    st.error(f"**Conclusion.** Sur {T} jours, le modèle enregistre {N} violations "
             f"pour {expected:.2f} attendues. Les tests mettent en évidence une "
             f"défaillance du modèle (zone {zone.lower()}). L'exigence en fonds "
             f"propres est portée à {money(FP)} avec un multiplicateur k = {k:.2f}.")

st.caption("Décision au seuil de 5 % : rejet de H₀ si la statistique LR dépasse "
           "la valeur critique du χ² (ou si la p-value < 0,05).")