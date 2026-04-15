# ============================================================
# TRADECLEANSE — NOTEBOOK 02 : Pipeline de Nettoyage Complet
# DCLE821 — QuantAxis Capital
# Etudiant(s) : Tom DERNONCOURT, Melchior DELESCLUSE, Sandra SINI
# Date        : 15 Avril 2026
# ============================================================
#
# CONTRAINTES OBLIGATOIRES :
#   - Ne jamais modifier tradecleanse_raw.csv
#   - Toujours travailler sur une copie : df = pd.read_csv(...).copy()
#   - Chaque etape doit etre loggee : nb lignes avant / apres / supprimees
#   - Chaque decision doit etre justifiee en commentaire (raison METIER)
#   - Le dataset final doit etre sauvegarde dans : tradecleanse_clean.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging (ne pas modifier)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('outputs/tradecleanse_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT (ne pas modifier)
# ============================================================
df_raw = pd.read_csv('data/tradecleanse_raw.csv', low_memory=False)
df = df_raw.copy()
logger.info(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================
# ETAPE 1 — Remplacement des valeurs sentinelles
# ============================================================
# CONSIGNE :
# Identifiez et remplacez TOUTES les valeurs sentinelles par NaN.
# Une sentinelle est une valeur utilisee a la place d'un NaN reel :
# textuelles (#N/A, N/A, #VALUE!, -, nd, null...) ET numeriques
# (ex: 99999 utilise comme code "donnee manquante" sur country_risk).
#
# ATTENTION : certaines colonnes sont en type "object" a cause des
# sentinelles textuelles melangees a des valeurs numeriques.
# Pensez a gerer le cast des colonnes apres nettoyage.
#
# Loggez le nb de NaN total avant et apres.

before = len(df)
nan_before = df.isnull().sum().sum()

TEXT_SENTINELS = ['N/A', '#N/A', '#VALUE!', '-', 'nd', 'null', 'NULL', 'nan', 'NaN', '']
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].replace(TEXT_SENTINELS, np.nan)

# 99999 sur country_risk : code "donnée manquante" Refinitiv, jamais un score valide
df['country_risk'] = df['country_risk'].replace(['99999', 99999], np.nan)
# 0.0 sur volatility_30d : utilise comme NaN Bloomberg (volatilite nulle impossible)
df['volatility_30d'] = df['volatility_30d'].replace(['0.0', '0'], np.nan)

nan_after = df.isnull().sum().sum()
logger.info(f"[Sentinelles] NaN avant={nan_before} -> apres={nan_after} (+{nan_after - nan_before} NaN crees)")
logger.info(f"[Sentinelles] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 2 — Suppression des doublons
# ============================================================
# CONSIGNE :
# Supprimez les doublons sur la cle metier trade_id.
# Justifiez dans un commentaire : pourquoi garder "first" ou "last" ?
# Dans le contexte Murex, quel enregistrement est le plus fiable ?
#
# Loggez : nb de doublons exacts, nb de doublons sur trade_id, shape finale.

before = len(df)
n_exact    = df.duplicated().sum()
n_trade_id = df.duplicated(subset='trade_id').sum()
logger.info(f"[Doublons] Doublons exacts={n_exact} | Doublons trade_id={n_trade_id}")

# keep='last' : dans Murex, la derniere occurrence d'un trade_id correspond
# a la version la plus recente apres eventuel amende (amendment) — c'est l'etat final fiable.
df = df.drop_duplicates(subset='trade_id', keep='last')
logger.info(f"[Doublons] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 3 — Conversion et normalisation des types
# ============================================================
# CONSIGNE :
# Convertissez chaque colonne vers son type pandas correct :
#   - trade_date, settlement_date : datetime (attention aux formats mixtes)
#   - bid, ask, mid_price, price, notional_eur,
#     quantity, volume_j, volatility_30d, country_risk : numerique
#   - asset_class, credit_rating, sector : chaine minuscule + strip
#
# Utilisez errors='coerce' pour les conversions — les valeurs non
# convertibles deviendront NaN (vous les traiterez a l'etape 8).
# Loggez le nb de valeurs devenues NaN par conversion rate.

before = len(df)
df['trade_date'] = pd.to_datetime(df['trade_date'],      errors='coerce')
df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')

for col in ['bid', 'ask', 'mid_price', 'price', 'notional_eur', 'quantity', 'volume_j', 'volatility_30d', 'country_risk']:
    nan_avant = df[col].isnull().sum()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    nan_apres = df[col].isnull().sum()
    if nan_apres > nan_avant:
        logger.info(f"[Types] {col} : {nan_apres - nan_avant} valeurs non convertibles -> NaN")

for col in ['asset_class', 'credit_rating', 'sector']:
    df[col] = df[col].astype(str).str.strip().str.lower().replace('nan', np.nan)

logger.info(f"[Types] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 4 — Normalisation du referentiel asset_class
# ============================================================
# CONSIGNE :
# La colonne asset_class contient de nombreuses variantes pour les
# 4 valeurs valides : equity, bond, derivative, fx.
# Construisez un dictionnaire de mapping exhaustif et appliquez-le.
# Toute valeur non mappee doit devenir NaN.
#
# Verifiez apres correction que seules les 4 valeurs existent.

before = len(df)
ASSET_CLASS_MAP = {
    'equity': 'equity', 'equities': 'equity', 'eq': 'equity',
    'bond': 'bond', 'fi': 'bond', 'fixed income': 'bond',
    'derivative': 'derivative', 'derivatives': 'derivative', 'deriv': 'derivative', 'opt': 'derivative',
    'fx': 'fx', 'forex': 'fx', 'foreign exchange': 'fx',
}
df['asset_class'] = df['asset_class'].map(ASSET_CLASS_MAP)
n_nan_ac = df['asset_class'].isnull().sum()
logger.info(f"[asset_class] {n_nan_ac} valeurs non mappees -> NaN | valeurs restantes : {df['asset_class'].value_counts().to_dict()}")
logger.info(f"[asset_class] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 5 — Incoherences structurelles financieres
# ============================================================
# CONSIGNE :
# Corrigez les 6 types d'incoherences metier suivants.
# Pour chacun, loggez le nb de lignes concernees ET justifiez
# la correction choisie (NaN ? Recalcul ? Valeur absolue ?).
#
# 5a. settlement_date < trade_date
#     Regle : le reglement intervient toujours apres le trade (T+2).
#     -> Quelle valeur mettre a la place ?
#
# 5b. bid > ask
#     Regle : la fourchette est toujours bid < ask.
#     -> Que faire des deux colonnes concernees ?
#
# 5c. mid_price incoherent avec (bid + ask) / 2
#     Regle : mid = (bid + ask) / 2, tolerance 1%.
#     -> Comment le recalculer ?
#
# 5d. price en dehors de la fourchette [bid * 0.995, ask * 1.005]
#     Regle : le prix d'execution ne peut pas etre hors fourchette.
#     -> Quelle valeur de substitution choisir ?
#
# 5e. notional_eur negatif
#     Regle : le notionnel est toujours positif pour une transaction standard.
#     -> Comment corriger sans perdre l'information ?
#
# 5f. credit_rating AAA/AA/A avec default_flag = 1
#     Regle : une contrepartie en defaut ne peut pas etre notee investissement.
#     -> Que faire du rating ? Que faire du flag ?

before = len(df)

bad_settle = df['settlement_date'] < df['trade_date']
df.loc[bad_settle, 'settlement_date'] = df.loc[bad_settle, 'trade_date'] + pd.Timedelta(days=2)
logger.info(f"[5a settlement] {bad_settle.sum()} lignes : settlement mis a trade_date+2j (regle T+2 marches actions)")

bad_bid_ask = df['bid'] > df['ask']
df.loc[bad_bid_ask, ['bid', 'ask']] = df.loc[bad_bid_ask, ['ask', 'bid']].values
logger.info(f"[5b bid/ask] {bad_bid_ask.sum()} lignes : bid et ask permutes (fourchette physiquement impossible inversee)")

mid_calc  = (df['bid'] + df['ask']) / 2
bad_mid   = (abs(df['mid_price'] - mid_calc) / mid_calc.replace(0, np.nan)) > 0.001
df.loc[bad_mid, 'mid_price'] = mid_calc[bad_mid].round(6)
logger.info(f"[5c mid_price] {bad_mid.sum()} lignes : mid_price recalcule = (bid+ask)/2 (erreur calcul source Bloomberg)")

bad_price = (df['price'] < df['bid'] * 0.995) | (df['price'] > df['ask'] * 1.005)
df.loc[bad_price, 'price'] = df.loc[bad_price, 'mid_price']
logger.info(f"[5d price] {bad_price.sum()} lignes : price hors fourchette -> remplace par mid_price (execution hors marche impossible)")

bad_notional = df['notional_eur'] < 0
df['notional_short_flag'] = 0
df.loc[bad_notional, 'notional_short_flag'] = 1
df.loc[bad_notional, 'notional_eur'] = df.loc[bad_notional, 'notional_eur'].abs()
logger.info(f"[5e notional] {bad_notional.sum()} lignes : notionnel negatif -> abs() + flag notional_short_flag=1 (preserve info position short)")

bad_rating = df['credit_rating'].isin(['aaa', 'aa', 'a']) & (df['default_flag'] == 1)
df['rating_degraded_due_to_default'] = 0
df.loc[bad_rating, 'credit_rating'] = 'ccc'
df.loc[bad_rating, "rating_degraded_due_to_default"] = 1
logger.info(f"[5f rating/defaut] {bad_rating.sum()} lignes : rating degrade a CCC (AAA/AA/A + defaut = contradiction logique impossible)")

logger.info(f"[Incoherences financieres] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 6 — Regles metier (valeurs hors plage valide)
# ============================================================
# CONSIGNE :
# Appliquez les regles metier suivantes colonne par colonne.
# Justifiez pour chaque regle si vous mettez NaN ou supprimez la ligne.
#
#   country_risk   : doit etre dans [0, 100]
#   volatility_30d : doit etre dans [0.1, 200]
#   default_flag   : doit etre 0 ou 1
#   quantity       : doit etre > 0

before = len(df)

cr_out = (df['country_risk'] < 0) | (df['country_risk'] > 100)
df.loc[cr_out, 'country_risk'] = np.nan
logger.info(f"[6 country_risk] {cr_out.sum()} hors [0,100] -> NaN (score pays ne peut depasser cette plage)")

vol_out = (df['volatility_30d'] <= 0) | (df['volatility_30d'] > 200)
df.loc[vol_out, 'volatility_30d'] = np.nan
logger.info(f"[6 volatility_30d] {vol_out.sum()} hors [0.1,200] -> NaN (volatilite negative ou >200% invraisemblable)")

df_flag_invalid = ~df['default_flag'].isin([0, 1])
df.loc[df_flag_invalid, 'default_flag'] = np.nan
logger.info(f"[6 default_flag] {df_flag_invalid.sum()} valeurs invalides -> NaN (binaire 0/1 uniquement)")

qty_out = df['quantity'] <= 0
df.loc[qty_out, 'quantity'] = np.nan
logger.info(f"[6 quantity] {qty_out.sum()} <= 0 -> NaN (quantite negative ou nulle n'a pas de sens metier)")

logger.info(f"[Regles metier] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 7 — Detection et traitement des outliers
# ============================================================
# CONSIGNE :
# Appliquez la methode IQR sur : notional_eur, volatility_30d, volume_j
#
# Pour chaque colonne :
#   1. Calculez Q1, Q3, IQR, lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
#   2. Comptez et affichez le nb d'outliers detectes
#   3. Choisissez une strategie (suppression / winsorisation / flaggage)
#      et justifiez-la en commentaire avec une raison METIER
#   4. Produisez un boxplot pour chaque colonne (avant traitement)
#
# Appliquez ensuite Isolation Forest sur les colonnes :
# [price, volume_j, volatility_30d, notional_eur]
# pour detecter les anomalies multivariees.
# Ajoutez une colonne "is_anomaly_multivariate" (0/1).
# Ne supprimez PAS ces lignes — le Risk Officer doit les examiner.
#
# LIBRAIRIES : scipy.stats, sklearn.ensemble.IsolationForest


# Winsorisation : on borne les valeurs extremes sans les supprimer.
# En finance, des notionnels ou volatilites extremes sont rares mais reels, les supprimer biaiserait le modele sur les queues de distribution

before = len(df)
from sklearn.ensemble import IsolationForest

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
outlier_cols = ['notional_eur', 'volatility_30d', 'volume_j']

for ax, col in zip(axes, outlier_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True, boxprops=dict(facecolor='#90caf9'))
    ax.set_title(f'{col} (avant)', fontsize=9)

plt.suptitle('Boxplots outliers IQR avant winsorisation', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_boxplots_outliers.png', dpi=120, bbox_inches='tight')
plt.close()

for col in outlier_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    logger.info(f"[7 IQR] {col}: {n_out} outliers winsories dans [{lower:.2f},{upper:.2f}]")

iso_features = ['price', 'volume_j', 'volatility_30d', 'notional_eur']
iso_df = df[iso_features].fillna(df[iso_features].median())
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
df['is_anomaly_multivariate'] = (iso.fit_predict(iso_df) == -1).astype(int)
n_anomaly = df['is_anomaly_multivariate'].sum()
logger.info(f"[7 IsolationForest] {n_anomaly} anomalies multivariées flagguees (conservées pour revue Risk Officer)")

logger.info(f"[Outliers] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 8 — Traitement des valeurs manquantes
# ============================================================
# CONSIGNE :
# Definissez une strategie par colonne. Regle generale :
#   < 20% NaN  : imputer (mediane pour numerique, mode pour categoriel)
#                + creer une colonne flag "colonne_was_missing" (0/1)
#   20%-70% NaN: imputer + flag (idem)
#   > 70% NaN  : supprimer la colonne
#
# Cas particuliers a justifier :
#   - settlement_date : quelle strategie pour les NaT ?
#   - credit_rating   : imputer le mode est-il pertinent pour un rating ?
#   - trade_id        : que faire si un trade_id est NaN ?
#
# Loggez la strategie choisie et le taux de NaN avant/apres pour chaque colonne.


before = len(df)

# trade_id NaN : un trade sans identifiant est intraitable — on ne peut pas
# le retrouver dans Murex, l auditer, ni le rattacher a une contrepartie.
# Suppression obligatoire, pas d imputation possible.

n_tid_nan = df['trade_id'].isnull().sum()
if n_tid_nan > 0:
    df = df.dropna(subset=['trade_id'])
    logger.info(f"[8 trade_id] {n_tid_nan} lignes supprimées : trade_id NaN = trade intraitable")

# settlement_date NaT : la regle T+2 est deterministe pour les actions.
# On calcule trade_date + 2j plutot que d imputer la mediane qui n aurait
# aucun sens metier ici (le delai de reglement n est pas aleatoire).

nat_settle = df['settlement_date'].isnull()
if nat_settle.sum() > 0:
    df.loc[nat_settle, 'settlement_date'] = df.loc[nat_settle, 'trade_date'] + pd.Timedelta(days=2)
    logger.info(f"[8 settlement_date] {nat_settle.sum()} NaT -> trade_date+2j (valeur deterministe T+2)")

# Colonnes numeriques : mediane + flag _was_missing.
# On choisit la mediane plutot que la moyenne car les distributions financieres
# sont asymetriques, la moyenne serait tirée par les grosses transactions.
# Le flag permet au modele de distinguer les vraies valeurs des imputées.
# Taux de NaN tous inferieurs a 20% apres traitement sentinelles/incoherences,
# donc on reste dans la regle generale imputer + flag.

for col in ['price', 'quantity', 'bid', 'ask', 'mid_price',
            'volume_j', 'notional_eur', 'country_risk', 'volatility_30d']:
    n_nan = df[col].isnull().sum()
    rate  = n_nan / len(df)
    if n_nan > 0:
        df[f'{col}_was_missing'] = df[col].isnull().astype(int)
        med = df[col].median()
        df[col] = df[col].fillna(med)
        logger.info(f"[8 {col}] {n_nan} NaN ({rate*100:.1f}%) -> mediane={med:.4f} + flag {col}_was_missing")

# asset_class et sector : mode + flag.
# La valeur la plus frequente est une imputation acceptable pour ces colonnes
# car elles ont peu de NaN et une distribution concentree sur quelques valeurs.

for col in ['asset_class', 'sector']:
    n_nan = df[col].isnull().sum()
    if n_nan > 0:
        df[f'{col}_was_missing'] = df[col].isnull().astype(int)
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        logger.info(f"[8 {col}] {n_nan} NaN -> mode='{mode_val}' + flag")

# credit_rating : on refuse d imputer le mode (BBB).
# Une contrepartie sans notation n'est pas une BBB  c'est une entité dont
# on ne connait pas le risque, ce qui est une information en soi.
# Imputer BBB introduirait un biais systematique vers le centre de la distribution
# et masquerait les contreparties les plus risquees du portefeuille.
# 'nr' (Not Rated) preserve cette distinction semantique pour le modele.

n_cr = df['credit_rating'].isnull().sum()
if n_cr > 0:
    df['credit_rating_was_missing'] = df['credit_rating'].isnull().astype(int)
    df['credit_rating'] = df['credit_rating'].fillna('nr')
    logger.info(f"[8 credit_rating] {n_cr} NaN -> 'nr' (Not Rated) : imputer le mode BBB serait trompeur pour un modele de risque)")

logger.info(f"[Valeurs manquantes] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 9 — Pseudonymisation RGPD / BCBS 239
# ============================================================
# CONSIGNE :
# Identifiez toutes les colonnes contenant des donnees PII
# (Personally Identifiable Information) ou des donnees sensibles.
#
# Pour chaque colonne PII :
#   1. Creez une colonne "colonne_hash" avec un hash SHA-256 irreversible
#   2. Supprimez la colonne originale
#
# Le salt doit etre lu depuis une variable d'environnement :
#   salt = os.environ.get('CLEANSE_SALT', 'default_salt_dev')
# Ne jamais hardcoder le salt dans le code.
#
# Justifiez dans un commentaire quelles colonnes sont des PII
# et pourquoi (reference a l'article RGPD correspondant).
#
# LIBRAIRIE : hashlib

# Les colonnes :
# counterparty_name et trader_id contiennent des informations d'identification directe (RGPD Art.4(1))
# qui permettent d'identifier une personne physique ou morale.
# Ces données sont sensibles car elles peuvent être utilisées pour retracer les transactions à des entités spécifiques, c
# ce qui pose un risque de confidentialité.
#  La pseudonymisation par hashage irreversible permet de protéger ces données
# tout en conservant la possibilité d'analyse au niveau agrégé sans exposer les identités réelles.

before = len(df)
import hashlib

salt = os.environ.get('CLEANSE_SALT', 'default_salt_dev')

for col in ['counterparty_name', 'trader_id']:
    df[f'{col}_hash'] = df[col].astype(str).apply(
        lambda x: hashlib.sha256(f"{salt}{x}".encode()).hexdigest()
    )
    df = df.drop(columns=[col])
    logger.info(f"[9 PII] '{col}' pseudonymise -> '{col}_hash' (SHA-256+salt). RGPD Art.4(5) : donnée identifiante indirecte)")

logger.info(f"[Pseudonymisation RGPD] {before} -> {len(df)} lignes")

# ============================================================
# ETAPE 10 — Rapport de qualite final
# ============================================================
# CONSIGNE :
# Produisez un rapport avant/apres comparant :
#   - Nb de lignes et colonnes
#   - Taux de completude global
#   - Nb de doublons restants
#   - Recapitulatif de chaque etape (nb lignes supprimees / modifiees)
#
# Calculez le Data Quality Score (DQS) selon la formule :
#   DQS = (completude * 0.6 + unicite * 0.4) * 100
#   ou completude = 1 - taux_nan_global
#   et unicite    = 1 - taux_doublons
#
# Sauvegardez le dataset nettoye.

nan_total  = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
completude  = 1 - nan_total / total_cells
unicite     = 1 - df.duplicated(subset='trade_id').mean()
dqs         = (completude * 0.6 + unicite * 0.4) * 100

print("\n" + "=" * 60)
print("RAPPORT QUALITE AVANT / APRES")
print("=" * 60)
print(f"{'Metrique':<35} {'Avant':>10} {'Apres':>10}")
print("-" * 60)
print(f"{'Nb lignes':<35} {df_raw.shape[0]:>10} {df.shape[0]:>10}")
print(f"{'Nb colonnes':<35} {df_raw.shape[1]:>10} {df.shape[1]:>10}")
raw_completude = (1 - df_raw.isnull().mean().mean()) * 100
print(f"{'Completude globale':<35} {raw_completude:>9.2f}% {completude*100:>9.2f}%")
print(f"{'Doublons trade_id':<35} {df_raw.duplicated(subset='trade_id').sum():>10} {df.duplicated(subset='trade_id').sum():>10}")
print(f"{'Data Quality Score (DQS)':<35} {'N/A':>10} {dqs:>9.2f}%")
print("=" * 60)
df.to_csv('data/tradecleanse_clean.csv', index=False)
logger.info(f"Dataset nettoyé et sauvegardé : data/tradecleanse_clean.csv")
print(f"Shape finale : {df.shape}")
