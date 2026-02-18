import pandas as pd

# Dictionnaire global de traduction
NOMS_FR = {
    "country_name": "Pays",
    "country_code": "Code ISO",
    "year": "Année",
    "life_evaluation_3_year_average": "Bonheur (moyenne 3 ans)",
    "predicted_happiness": "Bonheur prédit",

    # Variables macro
    "gdp_per_capita_current_usd": "PIB par habitant (USD)",
    "gdp_current_usd": "PIB total (USD)",
    "gross_national_income_usd": "Revenu national brut (USD)",
    "unemployment_rate_": "Taux de chômage (%)",
    "inflation_cpi": "Inflation (IPC)",
    "inflation_gdp_deflator": "Inflation (déflateur PIB)",
    "interest_rate_real": "Taux d’intérêt réel (%)",
    "gdp_growth_annual": "Croissance du PIB (%)",
    "current_account_balance_gdp": "Balance courante (% PIB)",
    "government_expense_of_gdp": "Dépenses publiques (% PIB)",
    "government_revenue_of_gdp": "Recettes publiques (% PIB)",
    "tax_revenue_of_gdp": "Recettes fiscales (% PIB)",
    "public_debt_of_gdp": "Dette publique (% PIB)",
}

def traduire_colonnes(df):
    return df.rename(columns=NOMS_FR)

def load_df():
    url = "https://aws-wh-bucket.s3.eu-north-1.amazonaws.com/final_clean.csv"
    df = pd.read_csv(url)
    return df