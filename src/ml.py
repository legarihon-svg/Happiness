import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# ---------------------------------------------------------
# 1. Construction X_var (comme dans 04_ML)
# ---------------------------------------------------------
def build_macro_matrix(df):
    cols_id = ["country_name", "country_code", "key_iso3_year", "year"]
    cols_leakage = [
        "lower_whisker", "upper_whisker",
        "explained_by_log_gdp_per_capita",
        "explained_by_social_support",
        "explained_by_healthy_life_expectancy",
        "explained_by_freedom_to_make_life_choices",
        "explained_by_generosity",
        "explained_by_perceptions_of_corruption",
        "dystopia_residual"
    ]
    cols_redundant = ["gdp_current_usd", "gross_national_income_usd"]

    target = "life_evaluation_3_year_average"

    X = df.drop(columns=cols_id + cols_leakage + cols_redundant + [target])

    imp = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    return X_clean, X_scaled


# ---------------------------------------------------------
# 2. Split temporel
# ---------------------------------------------------------
def split_train_test(df):
    df_sorted = df.sort_values("year")
    train = df_sorted[df_sorted["year"] <= 2018]
    test = df_sorted[df_sorted["year"] > 2018]
    return train, test


# ---------------------------------------------------------
# 3. Importances macro (comme dans 04_ML)
# ---------------------------------------------------------
def compute_wb_importances(df):
    target = "life_evaluation_3_year_average"
    X_clean, _ = build_macro_matrix(df)

    y = df[target]
    mask = ~y.isna()
    X_used = X_clean.loc[mask]
    y_used = y.loc[mask]

    model = RandomForestRegressor(random_state=42)
    model.fit(X_used, y_used)

    return pd.DataFrame({
        "feature": X_used.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


# ---------------------------------------------------------
# 4. Modèle final du bonheur
# ---------------------------------------------------------
def train_happiness_model(df):
    target = "life_evaluation_3_year_average"
    X_clean, _ = build_macro_matrix(df)
    features = X_clean.columns.tolist()

    train, test = split_train_test(df)

    X_train = train[features]
    y_train = train[target]

    median_hap = y_train.median()
    y_train_filled = y_train.fillna(median_hap)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0
    )

    model.fit(X_train, y_train_filled)
    return model, features


# ---------------------------------------------------------
# 5. Modèle macro générique
# ---------------------------------------------------------
def train_macro_model(train, test, target_macro, features_macro):
    X_train = train[features_macro]
    y_train = train[target_macro]

    median_y = y_train.median()
    y_train_filled = y_train.fillna(median_y)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0
    )

    model.fit(X_train, y_train_filled)
    return model


# ---------------------------------------------------------
# 6. Projection macro 2025–2030 (pays par pays)
# ---------------------------------------------------------
def project_macro_2030(df):
    X_clean, _ = build_macro_matrix(df)
    features_macro = X_clean.columns.tolist()

    train, test = split_train_test(df)

    projections = []

    for target_macro in features_macro:
        feats = [c for c in features_macro if c != target_macro]
        model = train_macro_model(train, test, target_macro, feats)

        for (cname, ccode), df_c in df.groupby(["country_name", "country_code"]):
            df_c_sorted = df_c.sort_values("year")
            last_row = df_c_sorted.iloc[-1]
            current_values = last_row[feats].copy()

            for year in range(2025, 2031):
                pred = model.predict(current_values.values.reshape(1, -1))[0]

                projections.append({
                    "country_name": cname,
                    "country_code": ccode,
                    "year": year,
                    "variable": target_macro,
                    "value": pred
                })

                if target_macro in current_values.index:
                    current_values[target_macro] = pred

    return pd.DataFrame(projections)


# ---------------------------------------------------------
# 7. Projection bonheur 2025–2030 (pays par pays)
# ---------------------------------------------------------
def project_happiness_2030(df):
    df_proj_macro = project_macro_2030(df)

    df_wide = df_proj_macro.pivot_table(
        index=["country_name", "country_code", "year"],
        columns="variable",
        values="value"
    ).reset_index()

    model_hap, features = train_happiness_model(df)

    df_wide["predicted_happiness"] = model_hap.predict(df_wide[features])

    return df_wide