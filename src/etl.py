"""
ETL complet World Happiness (WHR + World Bank) -> CSV final_clean.csv -> PostgreSQL (Neon)
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import country_converter as coco


# -----------------------------
# Config chemins & .env
# -----------------------------
load_dotenv()

PGHOST = os.getenv("PGHOST")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGPORT = os.getenv("PGPORT")

# À adapter à ton projet local
WHR_XLSX = "./data/raw/WHR25_Data_Figure_2.1v3.xlsx"
WB_CSV = "./data/raw/world_bank_data_2025.csv"
FINAL_CSV = "./data/processed/df_final.csv"
FINAL_CLEAN_CSV = "./data/processed/final_clean.csv"


# -----------------------------
# Helpers
# -----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


# -----------------------------
# 1. EXTRACT
# -----------------------------
def extract_whr_wb() -> tuple[pd.DataFrame, pd.DataFrame]:
    whr = pd.read_excel(WHR_XLSX)
    wb = pd.read_csv(WB_CSV)

    whr = clean_columns(whr)
    wb = clean_columns(wb)

    # ISO3
    whr["iso_3"] = coco.convert(names=whr["country_name"], to="ISO3")
    wb["iso_3"] = coco.convert(names=wb["country_name"], to="ISO3")

    # Supprimer Channel Islands (pas dans WHR)
    wb = wb[wb["country_name"] != "Channel Islands"].copy()

    # Refaire iso3 propre si besoin
    wb["iso3"] = coco.convert(names=wb["country_name"], to="ISO3")

    return whr, wb


# -----------------------------
# 2. TRANSFORM & MERGE
# -----------------------------
def build_df_final(whr: pd.DataFrame, wb: pd.DataFrame) -> pd.DataFrame:
    whr = whr.copy()
    wb = wb.copy()

    # Clé iso3_year
    whr["year"] = whr["year"].astype(str)
    wb["year"] = wb["year"].astype(str)
    whr["iso3_year"] = whr["iso_3"] + whr["year"]
    wb["iso3_year"] = wb["iso_3"] + wb["year"]

    # Supprimer North Cyprus & Somaliland Region (pas dans WB)
    whr = whr[
        (whr["country_name"] != "North Cyprus")
        & (whr["country_name"] != "Somaliland Region")
    ].copy()

    # Jointure inner sur iso3_year
    df_final = pd.merge(
        whr,
        wb,
        on="iso3_year",
        how="inner",
        suffixes=("_x", "_y"),
    )

    return df_final


def transform_to_final_clean(df_final: pd.DataFrame) -> pd.DataFrame:
    df = df_final.copy()

    # Drop colonnes inutiles
    cols_to_drop = [
        "rank",
        "iso_3_x",
        "iso_3_y",
        "country_name_y",
        "country_id",
        "year_y",
        "Unnamed: 0",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Renommer colonnes pour cohérence avec ton app + DB
    rename_map = {
        "year_x": "year",
        "country_name_x": "country_name",
        "iso3_year": "key_iso3_year",
        "inflation_cpi_": "inflation_cpi",
        "unemployment_rate_": "unemployment_rate_",
        "interest_rate_real_": "interest_rate_real",
        "inflation_gdp_deflator_": "inflation_gdp_deflator",
        "public_debt__of_gdp": "public_debt_of_gdp",
        "iso3": "country_code",
    }
    df = df.rename(columns=rename_map)

    # S'assurer de l'ordre des colonnes (comme dans ton notebook)
    ordered_cols = [
        "year",
        "country_name",
        "life_evaluation_3_year_average",
        "lower_whisker",
        "upper_whisker",
        "explained_by_log_gdp_per_capita",
        "explained_by_social_support",
        "explained_by_healthy_life_expectancy",
        "explained_by_freedom_to_make_life_choices",
        "explained_by_generosity",
        "explained_by_perceptions_of_corruption",
        "dystopia_residual",
        "key_iso3_year",
        "inflation_cpi",
        "gdp_current_usd",
        "gdp_per_capita_current_usd",
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
        "country_code",
    ]
    df = df[ordered_cols]

    # Types : objets -> string, arrondir floats
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("string[python]")

    num_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    df[num_cols] = df[num_cols].round(3)

    return df


# -----------------------------
# 3. LOAD TO CSV
# -----------------------------
def save_csvs(df_final: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(FINAL_CSV), exist_ok=True)
    df_final.to_csv(FINAL_CSV, index=False)
    df_clean.to_csv(FINAL_CLEAN_CSV, index=False)


# -----------------------------
# 4. LOAD TO POSTGRES (Neon)
# -----------------------------
def load_to_db(df_clean: pd.DataFrame) -> None:
    # Copie pour adapter aux noms de colonnes de la table
    df_db = df_clean.copy()

    # Harmoniser nom de colonne pour la DB
    df_db = df_db.rename(columns={"unemployment_rate_": "unemployment_rate"})

    # Nettoyage total (NaN -> None, etc.)
    df_db = df_db.replace(
        {
            np.nan: None,
            "nan": None,
            "NaN": None,
            "N/A": None,
            "": None,
            " ": None,
            "inf": None,
            "-inf": None,
        }
    )

    # Colonnes dans l'ordre exact de la table
    columns = [
        "year",
        "country_name",
        "life_evaluation_3_year_average",
        "lower_whisker",
        "upper_whisker",
        "explained_by_log_gdp_per_capita",
        "explained_by_social_support",
        "explained_by_healthy_life_expectancy",
        "explained_by_freedom_to_make_life_choices",
        "explained_by_generosity",
        "explained_by_perceptions_of_corruption",
        "dystopia_residual",
        "key_iso3_year",
        "inflation_cpi",
        "gdp_current_usd",
        "gdp_per_capita_current_usd",
        "unemployment_rate",
        "interest_rate_real",
        "inflation_gdp_deflator",
        "gdp_growth_annual",
        "current_account_balance_gdp",
        "government_expense_of_gdp",
        "government_revenue_of_gdp",
        "tax_revenue_of_gdp",
        "gross_national_income_usd",
        "public_debt_of_gdp",
        "country_code",
    ]

    values = [tuple(row[col] for col in columns) for _, row in df_db.iterrows()]

    conn = psycopg2.connect(
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        host=PGHOST,
        port=PGPORT,
        sslmode="require",
    )
    cursor = conn.cursor()

    insert_query = f"""
        INSERT INTO world_happiness ({", ".join(columns)})
        VALUES %s
    """

    execute_values(cursor, insert_query, values)
    conn.commit()
    cursor.close()
    conn.close()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print("1) Extraction WHR + World Bank...")
    whr, wb = extract_whr_wb()

    print("2) Jointure & df_final...")
    df_final = build_df_final(whr, wb)
    print("   df_final.shape =", df_final.shape)

    print("3) Transformation -> final_clean...")
    df_clean = transform_to_final_clean(df_final)
    print("   final_clean.shape =", df_clean.shape)

    print("4) Sauvegarde CSV...")
    save_csvs(df_final, df_clean)
    print(f"   Sauvegardé : {FINAL_CSV}")
    print(f"   Sauvegardé : {FINAL_CLEAN_CSV}")

    print("5) Chargement dans PostgreSQL (Neon)...")
    load_to_db(df_clean)
    print("   Import terminé avec succès !")


if __name__ == "__main__":
    main()