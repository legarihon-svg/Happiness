import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="World Happiness",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #6A1B9A; border-right: 1px solid #4A136E; }
h1,h2,h3 { color:#e6edf3; }
.section {
    border-left: 3px solid #58a6ff;
    padding-left: .7rem;
    margin:1.2rem 0 .6rem;
    color:#e6edf3;
    font-size: 1rem;
    font-weight:600;
}
.note {
    color:#8b949e;
    font-size: .82rem;
    margin-top: .3rem;
}
</style>
""",
    unsafe_allow_html=True,
)

WHR_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#111418",
    plot_bgcolor="#0D1117",
    font_color="#E6EDF3",
    margin=dict(l=10, r=10, t=55, b=10),
    colorway=["#005DAA", "#00A3A3", "#78C850", "#F2C94C", "#F2994A", "#EB5757"],
)

URL_RAW = "https://aws-wh-bucket.s3.eu-north-1.amazonaws.com/final_clean.csv"
URL_PROJ = "https://aws-wh-bucket.s3.eu-north-1.amazonaws.com/happiness_2011_2030.csv"
LABEL = "life_evaluation_3_year_average"

# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données ...")
def load_data():
    df_raw = pd.read_csv(URL_RAW)
    df_proj = pd.read_csv(URL_PROJ)
    df_proj["year"] = df_proj["year"].astype(int)
    return df_raw, df_proj.sort_values("year")


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Entraînement des modèles XGBoost ...")
def train_models(df):
    cols_id = ["country_name", "country_code", "key_iso3_year", "year"]
    cols_leak = [
        "lower_whisker",
        "upper_whisker",
        "explained_by_log_gdp_per_capita",
        "explained_by_social_support",
        "explained_by_healthy_life_expectancy",
        "explained_by_freedom_to_make_life_choices",
        "explained_by_generosity",
        "explained_by_perceptions_of_corruption",
        "dystopia_residual",
    ]
    cols_red = ["gdp_current_usd", "gross_national_income_usd"]

    y = df[LABEL]
    X = df.drop(columns=cols_id + cols_leak + cols_red + [LABEL], errors="ignore")

    imp_X = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp_X.fit_transform(X), columns=X.columns)

    sel = VarianceThreshold(threshold=1e-4)
    X_var = pd.DataFrame(sel.fit_transform(X_imp), columns=X_imp.columns[sel.get_support()])
    cols_kept = X_var.columns.tolist()

    # Temporal split
    train = df[df["year"] < 2020].copy()
    test = df[df["year"] >= 2020].copy()

    macro_features = [
        "inflation_cpi",
        "unemployment_rate_",
        "interest_rate_real",
        "inflation_gdp_deflator",
        "gdp_growth_annual",
        "current_account_balance_gdp",
        "government_expense_of_gdp",
        "government_revenue_of_gdp",
        "tax_revenue_of_gdp",
        "public_debt_of_gdp",
        "gdp_per_capita_current_usd",
    ]
    macro_targets = macro_features[:-1]
    gdp_feats = macro_targets

    # GDP model
    med_gdp = train["gdp_per_capita_current_usd"].median()
    model_gdp = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
    )
    model_gdp.fit(train[gdp_feats], train["gdp_per_capita_current_usd"].fillna(med_gdp))

    # Intermediate macro models
    macro_models = {}
    for tgt in macro_targets:
        feats = [c for c in macro_features if c != tgt]
        med = train[tgt].median()
        m = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=0,
        )
        m.fit(train[feats], train[tgt].fillna(med))
        macro_models[tgt] = m

    # Happiness model
    med_hap = train[LABEL].median()
    imp_hap = SimpleImputer(strategy="median")
    Xtr_hap = pd.DataFrame(imp_hap.fit_transform(train[cols_kept]), columns=cols_kept)

    model_hap = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
    )
    model_hap.fit(Xtr_hap, train[LABEL].fillna(med_hap))

    # Predict on all historical rows
    df2 = df.copy()
    Xall = pd.DataFrame(imp_hap.transform(df2[cols_kept].fillna(0)), columns=cols_kept)
    df2["predicted"] = model_hap.predict(Xall)

    # 2025–2030 projections
    future_rows = []
    for country in df["country_name"].unique():
        last = df[df["country_name"] == country].sort_values("year").iloc[-1].copy()
        for yr in range(2025, 2031):
            row = {"country_name": country, "year": yr}
            Xi = last[gdp_feats].values.reshape(1, -1)
            row["gdp_per_capita_current_usd"] = float(model_gdp.predict(Xi)[0])
            for tgt in macro_targets:
                feats = [c for c in macro_features if c != tgt]
                Xi = last[feats].values.reshape(1, -1)
                row[tgt] = float(macro_models[tgt].predict(Xi)[0])
            future_rows.append(row)
            last = pd.Series(row)

    df_fut = pd.DataFrame(future_rows)

    # Happiness on future rows
    fut_cols = [c for c in cols_kept if c in df_fut.columns]
    Xfut = df_fut.reindex(columns=cols_kept, fill_value=0)
    Xfut = pd.DataFrame(imp_hap.transform(Xfut), columns=cols_kept)
    df_fut["predicted"] = model_hap.predict(Xfut)

    # Bias correction + smoothing
    df_24 = df2[df2["year"] == 2024][["country_name", "predicted"]].rename(
        columns={"predicted": "pred_2024"}
    )
    real24 = df[df["year"] == 2024][["country_name", LABEL]].rename(
        columns={LABEL: "real_2024"}
    )
    df_fut = df_fut.merge(df_24, on="country_name", how="left")
    df_fut = df_fut.merge(real24, on="country_name", how="left")
    df_fut["offset"] = 0.5 * (df_fut["real_2024"] - df_fut["pred_2024"])
    df_fut[LABEL] = df_fut["predicted"] + df_fut["offset"].fillna(0)
    df_fut[LABEL] = (
        df_fut.groupby("country_name")[LABEL]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # Random Forest importance
    rf_feats = [
        c
        for c in [
            "inflation_cpi",
            "gdp_current_usd",
            "unemployment_rate_",
            "interest_rate_real",
            "inflation_gdp_deflator",
            "gdp_growth_annual",
            "current_account_balance_gdp",
            "government_expense_of_gdp",
            "government_revenue_of_gdp",
            "tax_revenue_of_gdp",
            "gross_national_income_usd",
            "public_debt_of_gdp",
        ]
        if c in df.columns
    ]
    df_ml = df[rf_feats + [LABEL]].dropna()
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(df_ml[rf_feats], df_ml[LABEL])
    imp_df = (
        pd.DataFrame({"variable": rf_feats, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    return df2, df_fut, imp_df, macro_features, macro_targets


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────

df_raw, df_proj = load_data()
df_hist, df_fut, imp_df, macro_features, macro_targets = train_models(df_raw)
all_countries = sorted(df_hist["country_name"].unique())

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## World Happiness")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📈  Prédit vs Réel", "🗺️  Carte mondiale", "📊  Variables macro"],
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#E6EDF3; font-size:.78rem;">'
        "XGBoost · Random Forest<br>World Bank · WHR 2011-2030"
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# PAGE 1 – Prédit vs Réel
# ─────────────────────────────────────────────────────────────

if page == "📈  Prédit vs Réel":
    st.title("📈 Score de bonheur — Prédit vs Réel")
    st.markdown(
        "Trait plein = **réel** · Pointillés = **modèle (2011–2030, sans 2024)**"
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.multiselect(
            "Pays",
            all_countries,
            default=["Finland", "France", "United States", "China", "Afghanistan"],
        )
    with c2:
        yr_range = st.slider("Période", 2011, 2030, (2011, 2030))

    if not selected:
        st.info("Sélectionnez au moins un pays.")
        st.stop()

    # On garde df_h / df_f pour la table récap
    df_h = (
        df_hist[df_hist["country_name"].isin(selected)][
            ["country_name", "year", LABEL, "predicted"]
        ]
        .rename(columns={LABEL: "réel", "predicted": "prédit"})
        .query("@yr_range[0] <= year <= @yr_range[1]")
    )

    df_f = (
        df_fut[df_fut["country_name"].isin(selected)][
            ["country_name", "year", LABEL]
        ]
        .rename(columns={LABEL: "proj"})
        .query("@yr_range[0] <= year <= @yr_range[1]")
    )

    palette = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, country in enumerate(selected):
        c = palette[i % len(palette)]

        # --- Réel 2011–2023 (on ignore 2024 pour éviter le pic) ---
        h_real = df_hist[
            (df_hist["country_name"] == country)
            & (df_hist["year"].between(yr_range[0], min(yr_range[1], 2023)))
        ].sort_values("year")

        # --- Modèle 2011–2023 (predicted) ---
        h_model_hist = df_hist[
            (df_hist["country_name"] == country)
            & (df_hist["year"].between(yr_range[0], min(yr_range[1], 2023)))
        ][["year", "predicted"]].sort_values("year")

        # --- Modèle 2025–2030 (df_fut[LABEL]) ---
        h_model_fut = df_fut[
            (df_fut["country_name"] == country)
            & (df_fut["year"].between(max(yr_range[0], 2025), yr_range[1]))
        ][["year", LABEL]].rename(columns={LABEL: "predicted"}).sort_values("year")

        # --- Modèle complet 2011–2030 (sans 2024) ---
        h_model = pd.concat([h_model_hist, h_model_fut], ignore_index=True)

        # --- Ligne réelle ---
        if not h_real.empty:
            fig.add_trace(
                go.Scatter(
                    x=h_real["year"],
                    y=h_real[LABEL],
                    mode="lines+markers",
                    name=f"{country} · réel",
                    line=dict(color=c, width=2.5),
                    marker=dict(size=5),
                    legendgroup=country,
                )
            )

        # --- Ligne modèle ---
        if not h_model.empty:
            fig.add_trace(
                go.Scatter(
                    x=h_model["year"],
                    y=h_model["predicted"],
                    mode="lines",
                    name=f"{country} · modèle",
                    line=dict(color=c, width=2, dash="dot"),
                    opacity=0.9,
                    legendgroup=country,
                )
            )

    fig.update_layout(
        **WHR_THEME,
        height=500,
        xaxis_title="Année",
        yaxis_title="Score de bonheur",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11, groupclick="toggleitem"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Table recap ───────────────────────────────────────────
    st.markdown(
        '<div class="section">Récapitulatif 2024 → 2030</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for country in selected:
        r24 = df_raw[
            (df_raw["country_name"] == country) & (df_raw["year"] == 2024)
        ][LABEL]
        r24 = float(r24.iloc[0]) if len(r24) else np.nan

        p30 = df_f[
            (df_f["country_name"] == country) & (df_f["year"] == 2030)
        ]["proj"]
        p30 = float(p30.iloc[0]) if len(p30) else np.nan

        rows.append(
            {
                "Pays": country,
                "Réel 2024": f"{r24:.3f}" if not np.isnan(r24) else "—",
                "Projection 2030": f"{p30:.3f}" if p30 and not np.isnan(p30) else "—",
                "Δ 2024→2030": f"{p30-r24:+.3f}"
                if (p30 and not np.isnan(p30) and not np.isnan(r24))
                else "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2 – Carte mondiale
# ─────────────────────────────────────────────────────────────

elif page == "🗺️  Carte mondiale":
    st.title("🗺️ Carte mondiale du bonheur")

    c1, c2 = st.columns([3, 1])
    with c1:
        year_sel = st.slider("Année", 2011, 2030, 2024)
    with c2:
        show_diff = st.checkbox("Variation vs année précédente", value=False)

    # Build year data
    if year_sel <= 2024:
        df_yr = df_proj[df_proj["year"] == year_sel][
            ["country_name", "year", LABEL]
        ].copy()
        src = "Données réelles / modélisées"
    else:
        df_yr = df_fut[df_fut["year"] == year_sel][
            ["country_name", "year", LABEL]
        ].copy()
        src = "Projection XGBoost"

    if show_diff and year_sel > 2011:
        prev = year_sel - 1
        df_pv_src = df_proj if prev <= 2024 else df_fut
        df_pv = df_pv_src[df_pv_src["year"] == prev][
            ["country_name", LABEL]
        ].rename(columns={LABEL: "prev"})
        df_yr = df_yr.merge(df_pv, on="country_name", how="left")
        df_yr["Δ bonheur"] = df_yr[LABEL] - df_yr["prev"]
        col_map, cscale, rng, cbar = (
            "Δ bonheur",
            ["#EB5757", "#F2994A", "#F2C94C", "#78C850", "#00A3A3"],
            [-0.4, 0.4],
            f"Δ vs {year_sel-1}",
        )
        title_map = f"Variation du bonheur : {year_sel} vs {year_sel-1}"
    else:
        col_map, cscale, rng, cbar = (
            LABEL,
            ["#EB5757", "#F2994A", "#F2C94C", "#78C850", "#00A3A3"],
            [2.5, 8.0],
            "Score",
        )
        title_map = f"Score de bonheur - {year_sel} ({src})"

    fig_map = px.choropleth(
        df_yr,
        locations="country_name",
        locationmode="country names",
        projection="orthographic",
        color=col_map,
        hover_name="country_name",
        hover_data={col_map: ":.3f", "year": False},
        color_continuous_scale=cscale,
        range_color=rng,
    )
    fig_map.update_layout(
        **WHR_THEME,
        height=560,
        title=title_map,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor="#1A1A1A",      # continents sombres
            showocean=True,
            oceancolor="#003F73",     # mer bleue
            showlakes=True,
            lakecolor="#003F73",
            resolution=50,
            projection=dict(
                type="orthographic",
                rotation=dict(lon=0, lat=0),
                scale=0.95
            ),
        ),
        coloraxis_colorbar=dict(title=cbar, ticks="outside"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top / Flop
    t1, t2 = st.tabs(["🏆 Top 10", "📉 Flop 10"])
    for tab, n, asc, cscale_b, label_tab in [
        (t1, 10, False, ["#4DA6FF", "#005DAA"], "Top 10"),
        (t2, 10, True, ["#EB5757", "#F2994A"], "Flop 10"),
    ]:
        with tab:
            if asc:
                # Flop 10 → croissant
                sub = df_yr.nsmallest(n, col_map).sort_values(col_map, ascending=True)
            else:
                # Top 10 → décroissant
                sub = df_yr.nlargest(n, col_map).sort_values(col_map, ascending=False)

            fig_b = px.bar(
                sub,
                x=col_map,
                y="country_name",
                orientation="h",
                color=col_map,
                color_continuous_scale=cscale_b,
                labels={col_map: cbar, "country_name": ""},
                text=sub[col_map].round(3).astype(str),
            )
            fig_b.update_traces(textposition="outside")
            fig_b.update_layout(**WHR_THEME, height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_b, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE 3 – Variables macro
# ─────────────────────────────────────────────────────────────

elif page == "📊  Variables macro":
    st.title("📊 Importance des variables macroéconomiques")
    st.markdown(
        "**Random Forest** sur données historiques + corrélations sur la projection 2025–2030."
    )

    VAR_LABELS = {
        "inflation_cpi": "Inflation (IPC)",
        "gdp_current_usd": "PIB total (USD)",
        "unemployment_rate_": "Taux de chômage",
        "interest_rate_real": "Taux d'intérêt réel",
        "inflation_gdp_deflator": "Déflateur du PIB",
        "gdp_growth_annual": "Croissance annuelle PIB",
        "current_account_balance_gdp": "Balance courante / PIB",
        "government_expense_of_gdp": "Dépenses publiques / PIB",
        "government_revenue_of_gdp": "Recettes publiques / PIB",
        "tax_revenue_of_gdp": "Recettes fiscales / PIB",
        "gross_national_income_usd": "Revenu national brut",
        "public_debt_of_gdp": "Dette publique / PIB",
    }

    df_imp = imp_df.copy()
    df_imp["pct"] = (df_imp["importance"] / df_imp["importance"].sum() * 100).round(1)
    df_imp["label"] = df_imp["variable"].map(VAR_LABELS).fillna(df_imp["variable"])

    # Importance bar
    st.markdown(
        '<div class="section">Importance relative - Random Forest (données historiques)</div>',
        unsafe_allow_html=True,
    )
    df_bar = df_imp.sort_values("pct")
    top5 = set(df_imp.nlargest(5, "pct")["variable"])
    colors = ["#00A3A3" if v in top5 else "#30363d" for v in df_bar["variable"]]
    fig_imp = go.Figure(
        go.Bar(
            x=df_bar["pct"],
            y=df_bar["label"],
            orientation="h",
            marker_color=colors,
            text=df_bar["pct"].astype(str) + "%",
            textposition="outside",
        )
    )
    fig_imp.update_layout(
        **WHR_THEME,
        height=450,
        xaxis_title="Importance (%)",
        yaxis_title="",
        xaxis=dict(range=[0, df_bar["pct"].max() * 1.25]),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Corrélation sur projection 2025–2030
    st.markdown(
        '<div class="section">Corrélation avec le bonheur projeté (2025–2030)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="note">Corrélation de Pearson entre chaque variable macro et le score de bonheur projeté.</p>',
        unsafe_allow_html=True,
    )

    proj_vars = [v for v in df_imp["variable"].tolist() if v in df_fut.columns]
    fut_25_30 = df_fut[df_fut["year"].between(2025, 2030)].copy()
    corr_rows = []
    for var in proj_vars:
        sub = fut_25_30[[var, LABEL]].dropna()
        if len(sub) > 10:
            corr_rows.append(
                {
                    "variable": var,
                    "label": VAR_LABELS.get(var, var),
                    "corr": sub[var].corr(sub[LABEL]),
                }
            )
    df_corr = pd.DataFrame(corr_rows).sort_values("corr")

    fig_corr = go.Figure(
        go.Bar(
            x=df_corr["corr"],
            y=df_corr["label"],
            orientation="h",
            marker_color=["#EB5757" if c < 0 else "#78C850" for c in df_corr["corr"]],
            text=df_corr["corr"].round(2).astype(str),
            textposition="outside",
        )
    )
    fig_corr.add_vline(x=0, line_color="#484f58")
    fig_corr.update_layout(
        **WHR_THEME,
        height=420,
        xaxis_title="Corrélation de Pearson",
        yaxis_title="",
        xaxis=dict(range=[-1.1, 1.1]),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter focus
    st.markdown(
        '<div class="section">Scatter - variable vs bonheur projeté (2025–2030)</div>',
        unsafe_allow_html=True,
    )

    var_options = [v for v in df_imp["variable"].tolist() if v in df_fut.columns]
    var_sel = st.selectbox(
        "Variable macroéconomique",
        var_options,
        format_func=lambda v: VAR_LABELS.get(v, v),
    )
    year_sc = st.select_slider(
        "Année (projection)", options=list(range(2025, 2031)), value=2027
    )

    df_sc = fut_25_30[fut_25_30["year"] == year_sc][
        ["country_name", var_sel, LABEL]
    ].dropna()

    fig_sc = px.scatter(
        df_sc,
        x=var_sel,
        y=LABEL,
        hover_name="country_name",
        color=LABEL,
        color_continuous_scale=["#EB5757", "#F2994A", "#F2C94C", "#78C850", "#00A3A3"],
        labels={var_sel: VAR_LABELS.get(var_sel, var_sel), LABEL: "Score de bonheur projeté"},
        trendline="ols",
        trendline_color_override="#4DA6FF",
        title=f"{VAR_LABELS.get(var_sel, var_sel)} vs Bonheur projeté - {year_sc}",
    )
    fig_sc.update_layout(**WHR_THEME, height=460, coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    # Tableau récapitulatif
    st.markdown(
        '<div class="section">Synthèse des 5 variables les plus influentes</div>',
        unsafe_allow_html=True,
    )
    top5_df = df_imp.head(5)[["label", "pct"]].copy()
    top5_df.columns = ["Variable", "Importance (%)"]
    interpretations = [
        "Richesse absolue → levier principal sur le bonheur",
        "Sécurité économique des ménages",
        "Capacité de financement des services publics",
        "Richesse nationale globale disponible",
        "Stabilité macroéconomique perçue",
    ]
    top5_df["Interprétation"] = interpretations[: len(top5_df)]
    st.dataframe(top5_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.markdown(
    '<div style="text-align:center; color:#9BA3B0; font-size:.72rem; '
    'margin-top:2rem; padding-top:1rem; border-top:1px solid #30363d">'
    "World Happiness Predictor · XGBoost + Random Forest · World Bank & WHR · 2011-2030"
    "</div>",
    unsafe_allow_html=True,
)