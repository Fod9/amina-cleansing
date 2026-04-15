# ============================================================
# TRADECLEANSE — NOTEBOOK 03 : Validation du Dataset Nettoye
# DCLE821 — QuantAxis Capital
# Etudiant(s) : Tom DERNONCOURT, Melchior DELESCLUSE, Sandra SINI
# Date        : 15 Avril 2026
# ============================================================
#
# CONSIGNE GENERALE :
# Ce notebook valide que votre pipeline a correctement nettoye le dataset.
# Vous devez implementer au minimum 14 tests de validation (expectations).
#
# Deux approches possibles :
#   A) Utiliser la librairie Great Expectations (recommande en entreprise)
#      pip install great_expectations
#      Documentation : https://docs.greatexpectations.io
#
#   B) Implementer vos propres tests avec pandas + assertions Python
#      (acceptable si vous documentez clairement chaque test)
#
# Pour chaque test, affichez clairement : [PASS] ou [FAIL] + le detail.
# A la fin, affichez un score : X/14 tests passes.
# ============================================================

import pandas as pd
import numpy as np
import great_expectations as gx
import warnings

warnings.filterwarnings('ignore')

# Chargement du dataset nettoye
# ATTENTION : ce fichier doit avoir ete genere par 02_cleaning_pipeline.py
df = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)

df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')

# Variables pré-calculées pour valider les expectations complexes
df['price_valid'] = (df['price'] >= df['bid'] * 0.995) & (df['price'] <= df['ask'] * 1.005)
mid_theo = (df['bid'] + df['ask']) / 2
df['mid_price_valid'] = (abs(df['mid_price'] - mid_theo) / mid_theo.replace(0, np.nan)) <= 0.001
df['rating_valid'] = ~(df['credit_rating'].isin(['aaa', 'aa', 'a']) & (df['default_flag'] == 1))
nan_rate = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
df['global_completeness_valid'] = nan_rate < 0.10
df['pii_absent'] = ('counterparty_name' not in df.columns) and ('trader_id' not in df.columns)

# Initialisation du contexte Great Expectations v1.0
context = gx.get_context()
data_source = context.data_sources.add_pandas("pandas_source")
data_asset = data_source.add_dataframe_asset("trade_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe("trade_batch")

suite = gx.ExpectationSuite(name="trade_suite")

# ============================================================
# Validation du dataset nettoye : (14 expectations attendues)
# ============================================================

# ============================================================
# EXPECTATION 1 — Unicite de trade_id
# ============================================================
# Verifie qu'il n'existe aucun doublon sur la cle metier.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="trade_id"))

# ============================================================
# EXPECTATION 2 — Colonnes obligatoires non nulles
# ============================================================
# Les colonnes suivantes ne doivent contenir aucun NaN :
# trade_id, counterparty_id, isin, trade_date,
# asset_class, price, quantity, default_flag
for col in ['trade_id', 'counterparty_id', 'isin', 'trade_date', 'asset_class', 'price', 'quantity', 'default_flag']:
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

# ============================================================
# EXPECTATION 3 — settlement_date >= trade_date
# ============================================================
# Un reglement ne peut pas etre anterieur au trade.
suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
    column_A="settlement_date",
    column_B="trade_date",
    or_equal=True
))

# ============================================================
# EXPECTATION 4 — bid < ask sur toutes les lignes
# ============================================================
# La fourchette de prix doit toujours etre dans le bon sens.
suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="ask", column_B="bid"))

# ============================================================
# EXPECTATION 5 — price dans la fourchette [bid * 0.995, ask * 1.005]
# ============================================================
# Un prix d'execution ne peut pas etre en dehors de la fourchette.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="price_valid", value_set=[True]))

# ============================================================
# EXPECTATION 6 — mid_price coherent avec (bid + ask) / 2
# ============================================================
# Tolerance : ecart < 1% du mid theorique.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="mid_price_valid", value_set=[True]))

# ============================================================
# EXPECTATION 7 — asset_class dans le referentiel normalise
# ============================================================
# Seules ces valeurs sont acceptees : equity, bond, derivative, fx
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="asset_class", value_set=['equity', 'bond', 'derivative', 'fx']))

# ============================================================
# EXPECTATION 8 — Pas de contradiction rating investissement + defaut
# ============================================================
# credit_rating AAA, AA ou A avec default_flag = 1 est impossible.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="rating_valid", value_set=[True]))

# ============================================================
# EXPECTATION 9 — notional_eur > 0
# ============================================================
# Le montant notionnel doit etre strictement positif.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="notional_eur", min_value=0, strict_min=True))

# ============================================================
# EXPECTATION 10 — country_risk dans [0, 100]
# ============================================================
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="country_risk", min_value=0, max_value=100))

# ============================================================
# EXPECTATION 11 — Format ISIN valide
# ============================================================
# Un ISIN est compose de 2 lettres majuscules + 10 caracteres alphanumeriques.
# Regex : ^[A-Z]{2}[A-Z0-9]{10}$
suite.add_expectation(gx.expectations.ExpectColumnValuesToMatchRegex(column="isin", regex=r"^[A-Z]{2}[A-Z0-9]{10}$"))

# ============================================================
# EXPECTATION 12 — volatility_30d dans [0.1, 200]
# ============================================================
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="volatility_30d", min_value=0.1, max_value=200))

# ============================================================
# EXPECTATION 13 — Completude globale > 90%
# ============================================================
# Le taux de completude global (toutes colonnes confondues) doit
# etre superieur a 90%.
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="global_completeness_valid", value_set=[True]))

# ============================================================
# EXPECTATION 14 — Absence de PII en clair
# ============================================================
# Les colonnes counterparty_name et trader_id ne doivent PAS
# exister dans le dataset final (remplacees par leurs versions hashees).
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="pii_absent", value_set=[True]))

# Enregistrement et exécution de la suite
suite = context.suites.add(suite)

validation_definition = gx.ValidationDefinition(
    data=batch_definition,
    suite=suite,
    name="trade_validation"
)
validation_definition = context.validation_definitions.add(validation_definition)

validation_results = validation_definition.run(batch_parameters={"dataframe": df})

# Formatage des résultats
results_list = []
passed_count = 0
total_count = 0

for res in validation_results.results:
    total_count += 1
    if res.success:
        passed_count += 1

    results_list.append({
        'expectation': res.expectation_config.type,
        'kwargs': str(res.expectation_config.kwargs),
        'status': 'PASS' if res.success else 'FAIL'
    })

# ============================================================
# SCORE FINAL
# ============================================================
# Affichez : "Score : X/14 expectations passees"
# Exportez les resultats dans un fichier ge_validation_report.csv
print("=" * 80)
print(f"Score : {passed_count}/{total_count} expectations passées")
print("=" * 80)

pd.DataFrame(results_list).to_csv('outputs/03_ge_validation_report.csv', index=False)
