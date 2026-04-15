# ============================================================
# TRADECLEANSE — NOTEBOOK 01 : Audit & Profiling Initial
# DCLE821 — QuantAxis Capital
# Etudiant(s) : ___________________________________
# Date        : ___________________________________
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CELLULE 1 — Chargement multi-sources
# ============================================================
# CONSIGNE :
# Le dataset consolide 3 sources heterogenes :
#   - Bloomberg   : fichier CSV (colonnes marche : bid, ask, mid_price, price,
#                                volume_j, volatility_30d)
#   - Murex (SQL) : transactions internes (trade_id, dates, notional, quantity,
#                                          trader_id, asset_class)
#   - Refinitiv   : donnees fondamentales (credit_rating, sector,
#                                          counterparty_name, country_risk)
#
# TACHE :
#   1. Chargez tradecleanse_raw.csv avec les parametres d'import appropries.
#      Pensez a gerer : encodage, separateur, valeurs sentinelles connues,
#      types de colonnes.
#   2. Simulez les 3 sources en creant 3 sous-dataframes avec uniquement
#      les colonnes correspondant a chaque source.
#   3. Ajoutez une colonne "source" sur chacun avant consolidation.
#
# LIBRAIRIES SUGGÉREES : pandas, sqlite3 (pour simuler Murex en SQL)


df_raw = pd.read_csv('data/tradecleanse_raw.csv', low_memory=False)
df = df_raw.copy()

SENTINELS = ['N/A', '#N/A', '#VALUE!', '-', 'nd', 'null', 'NULL', 'nan', 'NaN', '']

MUREX_COLS     = ['trade_id', 'counterparty_id', 'trade_date', 'settlement_date',
                  'asset_class', 'notional_eur', 'quantity', 'trader_id']
BLOOMBERG_COLS = ['trade_id', 'isin', 'price', 'bid', 'ask', 'mid_price', 'volume_j', 'volatility_30d']
REFINITIV_COLS = ['trade_id', 'counterparty_name', 'credit_rating', 'default_flag', 'sector', 'country_risk']

# Murex simule via SQLite (source originale = base transactionnelle SQL)
conn = sqlite3.connect(':memory:')
df_raw[MUREX_COLS].to_sql('murex_trades', conn, index=False, if_exists='replace')
df_murex = pd.read_sql('SELECT * FROM murex_trades', conn)
df_murex['source'] = 'murex'
conn.close()

df_bloomberg = df_raw[BLOOMBERG_COLS].copy()
df_bloomberg['source'] = 'bloomberg'

df_refinitiv = df_raw[REFINITIV_COLS].copy()
df_refinitiv['source'] = 'refinitiv'

print(f"Dataset consolide : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"Murex : {len(df_murex)} transactions")
print(f"Bloomberg : {len(df_bloomberg)} lignes de market data")
print(f"Refinitiv : {len(df_refinitiv)} fiches contrepartie")


# ============================================================
# CELLULE 2 — Profiling initial
# ============================================================
# CONSIGNE :
# Produisez un rapport de profiling complet du dataset brut.
# Il doit contenir au minimum :
#   - Shape (nb lignes, nb colonnes)
#   - Types detectes par pandas
#   - Taux de valeurs manquantes par colonne (count + %)
#   - Statistiques descriptives (min, max, mean, std, quartiles)
#     pour toutes les colonnes numeriques
#   - Cardinalite (nb de valeurs uniques) pour les colonnes categorielles
#   - Distribution de chaque colonne categorielle (value_counts)
#   - Nombre de doublons exacts et doublons sur trade_id
#
# LIBRAIRIES SUGGÉREES : pandas (.describe, .isnull, .value_counts, .duplicated)

print("\n" + "="*65)
print("Profiling des données")
print("="*65)

print("\nTypes pandas")
print(df.dtypes.to_string())

nan_nat = df.isnull().sum()
print("\nNaN natifs par colonne")
print(nan_nat[nan_nat > 0].to_frame('count')
      .assign(pct=lambda x: (x['count'] / len(df) * 100).round(2))
      .to_string())


print("\nSentinelles textuelles masquant des NaN réels")
for col in df.select_dtypes(include='object').columns:
    hits = {s: int((df[col] == s).sum()) for s in SENTINELS if (df[col] == s).sum() > 0}
    if hits:
        print(f"   {col:20s} : {hits}")

n_99999 = int((df['country_risk'] == '99999').sum())
print(f"   {'country_risk':20s} : {{'99999': {n_99999}}} sentinel numerique Refinitiv")

print("\nManquants réels totaux par colonne")
missing_real = nan_nat.copy()
for col in df.select_dtypes(include='object').columns:
    missing_real[col] += int(df[col].isin(SENTINELS).sum())
missing_real['country_risk'] += n_99999
missing_real = missing_real[missing_real > 0].sort_values(ascending=False)
print(missing_real.to_frame('total_missing')
      .assign(pct=lambda x: (x['total_missing'] / len(df) * 100).round(1))
      .to_string())

num_cols = ['notional_eur', 'price', 'quantity', 'bid', 'ask', 'mid_price', 'volume_j']
print("\nStatistiques descriptives colonnes numeriques")
print(df[num_cols].describe().round(2).to_string())

print("\nColonnes object a contenu numerique (apres cast)")
for col in ['volatility_30d', 'country_risk']:
    s = pd.to_numeric(df[col], errors='coerce')
    print(f"   {col}: min={s.min():.2f}  max={s.max():.2f}  "
          f"mean={s.mean():.2f}  NaN post-cast={s.isnull().sum()}")

print("\nVariables categorielles")
for col in ['asset_class', 'credit_rating', 'sector', 'default_flag']:
    vc = df[col].value_counts(dropna=False)
    print(f"\n   {col}  ({df[col].nunique(dropna=True)} valeurs distinctes)")
    print(vc.to_string())

print(f"\nDoublons")
print(f"Doublons ligne complete : {df.duplicated().sum()}")
print(f"Doublons sur trade_id   : {df.duplicated(subset='trade_id').sum()}  <- migration Murex")

print("\nCorrelations bid / ask / mid_price / price")
print(df[['bid', 'ask', 'mid_price', 'price']].corr().round(4).to_string())

mid_theo = (df['bid'] + df['ask']) / 2
ecart_mid = (df['mid_price'] - mid_theo).abs()
print(f"\n   Lignes ou |mid_price - (bid+ask)/2| > 1% : "
      f"{int((ecart_mid / mid_theo.replace(0, np.nan) > 0.01).sum())}")

print("\nOutliers IQR (1.5*IQR)")
for col in ['notional_eur', 'price', 'bid', 'ask', 'volume_j']:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_out = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    print(f"{col:15s} Q1={q1:>12.2f} Q3={q3:>12.2f}  "
          f"IQR={iqr:>12.2f} outliers={n_out} ({n_out/len(s)*100:.1f}%)")

td = pd.to_datetime(df['trade_date'],      errors='coerce')
sd = pd.to_datetime(df['settlement_date'], errors='coerce')
delta = (sd - td).dt.days
print(f"\nAnalyse temporelle")
print(f"trade_date  : {td.min().date()}  ->  {td.max().date()}")
print(f"settlement  : {sd.min().date()}  ->  {sd.max().date()}")
print(f"Delai moyen settlement - trade : {delta.mean():.2f} jours")
print(f"Lignes avec settlement < trade : {int((delta < 0).sum())}  <- viole T+2")


# ============================================================
# CELLULE 3 — Detection des anomalies
# ============================================================
# CONSIGNE :
# A partir du profiling, identifiez et quantifiez chaque anomalie.
# Pour chaque anomalie trouvee, vous devez indiquer :
#   - Le type (doublon / valeur manquante / outlier / incoherence / format...)
#   - La ou les colonnes concernees
#   - Le nombre de lignes impactees
#   - La criticite metier (impact sur le modele de risque)
#
# Construisez un dictionnaire ou DataFrame "anomalies_report" qui recense
# tout ce que vous avez trouve.
#
# RAPPEL DES COLONNES ET LEURS REGLES METIER :
#   trade_id          : doit etre unique
#   settlement_date   : doit etre >= trade_date (regle T+2 actions)
#   bid / ask         : bid doit toujours etre < ask
#   mid_price         : doit etre egal a (bid + ask) / 2
#   price             : doit se trouver dans la fourchette [bid, ask] +/- 0.5%
#   notional_eur      : doit etre positif (sauf position short documentee)
#   asset_class       : doit appartenir a {equity, bond, derivative, fx}
#   credit_rating     : valeurs valides AAA AA A BBB BB B CCC D
#   country_risk      : doit etre compris entre 0 et 100
#   volatility_30d    : doit etre > 0 et < 200
#   default_flag      : valeurs valides 0 ou 1 uniquement
#   credit_rating + default_flag : un emetteur note AAA/AA/A ne peut pas
#                                  avoir default_flag = 1 (contradiction)


vol_num  = pd.to_numeric(df['volatility_30d'], errors='coerce')
cr_num = pd.to_numeric(df['country_risk'],   errors='coerce')
td = pd.to_datetime(df['trade_date'],     errors='coerce')
sd = pd.to_datetime(df['settlement_date'],errors='coerce')
mid_theo = (df['bid'] + df['ask']) / 2

anomalies_report = pd.DataFrame([
    {
        'type': 'Doublon',
        'colonne': 'trade_id',
        'count': int(df.duplicated(subset='trade_id').sum()),
        'criticite': 'HAUTE',
        'description': 'Doublons issus de la migration Murex cle metier non unique'
    },
    {
        'type': 'Sentinelle textuelle',
        'colonne': 'volatility_30d',
        'count': int(df['volatility_30d'].isin(['-', '#VALUE!']).sum()),
        'criticite': 'MOYENNE',
        'description': 'Export Bloomberg produit #VALUE! et - quand la vol est indisponible'
    },
    {
        'type': 'Sentinelle numerique',
        'colonne': 'country_risk',
        'count': int((df['country_risk'] == '99999').sum()),
        'criticite': 'MOYENNE',
        'description': 'Refinitiv utilise 99999 comme code absence de donnée hors plage [0,100]'
    },
    {
        'type': 'Valeur manquante',
        'colonne': 'credit_rating',
        'count': int(df['credit_rating'].isnull().sum()),
        'criticite': 'HAUTE',
        'description': '~15% NaN natifs colonne determinante pour le scoring de risque'
    },
    {
        'type': 'Valeur manquante',
        'colonne': 'volatility_30d',
        'count': int(df['volatility_30d'].isnull().sum() + df['volatility_30d'].isin(['-', '#VALUE!']).sum()),
        'criticite': 'MOYENNE',
        'description': 'Manquants réels = NaN natifs + sentinelles Bloomberg'
    },
    {
        'type': 'Casse inconsistante',
        'colonne': 'asset_class',
        'count': int(df['asset_class'].nunique()),
        'criticite': 'HAUTE',
        'description': '20 variantes textuelles pour 4 valeurs valides (equity/bond/derivative/fx)'
    },
    {
        'type': 'Incoherence metier',
        'colonne': 'settlement_date',
        'count': int((sd < td).sum()),
        'criticite': 'CRITIQUE',
        'description': 'settlement_date < trade_date impossible viole la regle T+2'
    },
    {
        'type': 'Incoherence metier',
        'colonne': 'bid / ask',
        'count': int((df['bid'] > df['ask']).sum()),
        'criticite': 'CRITIQUE',
        'description': 'bid > ask : fourchette physiquement impossible sur un marche organise'
    },
    {
        'type': 'Incoherence calcul',
        'colonne': 'mid_price',
        'count': int(((df['mid_price'] - mid_theo).abs() / mid_theo.replace(0, np.nan) > 0.001).sum()),
        'criticite': 'HAUTE',
        'description': 'mid_price != (bid+ask)/2 : erreur de calcul a la source Bloomberg'
    },
    {
        'type': 'Incoherence metier',
        'colonne': 'price',
        'count': int(((df['price'] < df['bid'] * 0.995) | (df['price'] > df['ask'] * 1.005)).sum()),
        'criticite': 'CRITIQUE',
        'description': 'Prix execution hors fourchette bid/ask +/-0.5% : execution impossible'
    },
    {
        'type': 'Valeur negative',
        'colonne': 'notional_eur',
        'count': int((df['notional_eur'] < 0).sum()),
        'criticite': 'HAUTE',
        'description': 'Notionnel negatif sans position short documentée'
    },
    {
        'type': 'Contradiction',
        'colonne': 'credit_rating + default_flag',
        'count': int((df['credit_rating'].isin(['AAA', 'AA', 'A']) & (df['default_flag'] == 1)).sum()),
        'criticite': 'CRITIQUE',
        'description': 'Emetteur note investment grade ET en defaut : logiquement impossible'
    },
    {
        'type': 'Hors plage',
        'colonne': 'country_risk',
        'count': int(((cr_num < 0) | (cr_num > 100)).sum()),
        'criticite': 'MOYENNE',
        'description': 'Score risque pays hors [0, 100]'
    },
    {
        'type': 'Hors plage',
        'colonne': 'volatility_30d',
        'count': int(((vol_num <= 0) | (vol_num > 200)).sum()),
        'criticite': 'MOYENNE',
        'description': 'Volatilite negative ou > 200% hors plage physiquement credible'
    },
])

print("\nAnomalies detectes")
print(anomalies_report.to_string(index=False))
print(f"\n{len(anomalies_report)} types | "
      f"{anomalies_report['count'].sum()} occurrences sur {len(df)} lignes")


# ============================================================
# CELLULE 4 — Visualisations
# ============================================================
# CONSIGNE :
# Produisez au minimum 4 graphiques illustrant les anomalies detectees :
#   - Taux de valeurs manquantes par colonne (barh)
#   - Distribution de asset_class (toutes variantes)
#   - Scatter bid vs ask (mettre en evidence les inversions)
#   - Distribution du delai settlement - trade_date (histogramme)
#
# Sauvegardez le tout dans un seul fichier : 01_profiling_report.png

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1    = fig.add_subplot(gs[0, 0])
colors = ['#d32f2f' if v > 500 else '#f57c00' if v > 100 else '#388e3c'
          for v in missing_real.values]
bars   = ax1.barh(missing_real.index, missing_real.values, color=colors)
for bar, val in zip(bars, missing_real.values):
    ax1.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
             f'{val}  ({val/len(df)*100:.1f}%)', va='center', fontsize=8)
ax1.set_xlim(0, missing_real.max() * 1.4)
ax1.set_xlabel('Occurrences')
ax1.set_title('Valeurs manquantes réelles\n(NaN natifs + sentinelles)', fontweight='bold')

ax2   = fig.add_subplot(gs[0, 1])
ac_vc = df['asset_class'].value_counts(dropna=False)
ax2.bar(range(len(ac_vc)), ac_vc.values,
        color=plt.cm.tab20(np.linspace(0, 1, len(ac_vc))))
ax2.set_xticks(range(len(ac_vc)))
ax2.set_xticklabels(ac_vc.index.astype(str), rotation=45, ha='right', fontsize=7)
ax2.set_title('Distribution asset_class\n20 variantes pour 4 valeurs valides', fontweight='bold')
ax2.set_ylabel('Count')

ax3     = fig.add_subplot(gs[1, 0])
bid_s   = df['bid']
ask_s   = df['ask']
ok_idx  = bid_s[bid_s <= ask_s].sample(min(600, (bid_s <= ask_s).sum()), random_state=42).index
bad_idx = bid_s[bid_s > ask_s].index
ax3.scatter(bid_s[ok_idx],  ask_s[ok_idx],  alpha=0.25, s=5,
            color='#1976d2', label=f'bid <= ask  (n={(bid_s<=ask_s).sum():,})')
ax3.scatter(bid_s[bad_idx], ask_s[bad_idx], alpha=0.8,  s=18,
            color='#d32f2f', label=f'bid > ask   (n={len(bad_idx)} ANOMALIE)')
lim = [min(bid_s.min(), ask_s.min()), max(bid_s.max(), ask_s.max())]
ax3.plot(lim, lim, 'k--', lw=1, alpha=0.5)
ax3.set_xlabel('bid')
ax3.set_ylabel('ask')
ax3.set_title('Scatter bid vs ask\nInversions de fourchette en rouge', fontweight='bold')
ax3.legend(fontsize=8)

ax4   = fig.add_subplot(gs[1, 1])
delta = (pd.to_datetime(df['settlement_date'], errors='coerce') -
         pd.to_datetime(df['trade_date'],       errors='coerce')).dt.days.dropna()
ax4.hist(delta[delta >= -10], bins=35, color='#5c6bc0', edgecolor='white', lw=0.4)
ax4.axvline(0, color='#d32f2f', lw=2, ls='--')
ax4.set_xlabel('settlement_date - trade_date  (jours)')
ax4.set_ylabel('Frequence')
ax4.set_title('Delai settlement - trade\nLes valeurs < 0 violent la regle T+2', fontweight='bold')
ax4.text(0.03, 0.93, f'{int((delta < 0).sum())} lignes avec delta < 0',
         transform=ax4.transAxes, color='#d32f2f', fontsize=9,
         bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax5       = fig.add_subplot(gs[2, 0])
corr_cols = ['bid', 'ask', 'mid_price', 'price', 'notional_eur', 'volume_j']
corr      = df[corr_cols].corr()
im        = ax5.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax5.set_xticks(range(len(corr_cols)))
ax5.set_xticklabels(corr_cols, rotation=40, ha='right', fontsize=8)
ax5.set_yticks(range(len(corr_cols)))
ax5.set_yticklabels(corr_cols, fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax5.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                 fontsize=7, color='black')
plt.colorbar(im, ax=ax5, shrink=0.75)
ax5.set_title('Heatmap correlations\nbid / ask / mid / price', fontweight='bold')

ax6 = fig.add_subplot(gs[2, 1])
ct  = pd.crosstab(df['credit_rating'], df['default_flag'])
ct  = ct.reindex(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D'], fill_value=0)
ct.plot(kind='bar', ax=ax6, color=['#43a047', '#e53935'], edgecolor='white', width=0.7)
ax6.set_xlabel('credit_rating')
ax6.set_ylabel('Count')
ax6.set_title('credit_rating x default_flag\nBarres rouges sur AAA/AA/A = contradictions',
              fontweight='bold')
ax6.legend(['sain (0)', 'defaut (1)'], fontsize=8)
ax6.tick_params(axis='x', rotation=0)

fig.suptitle('TradeCleanse Rapport de Profiling Initial | QuantAxis Capital',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig('outputs/01_profiling_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n01_profiling_report.png sauvegardé dans outputs/")
