import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
import hashlib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')


class DataCleaner:

    def __init__(self, config_path: str = "config/cleaning.json"):
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self.df_raw = None
        self.df = None
        self.logger = None

    def setup_logging(self):
        log_path = self.config["io"]["log_path"]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load(self):
        io = self.config["io"]
        self.df_raw = pd.read_csv(io["input_path"], low_memory=False)
        self.df = self.df_raw.copy()
        self.logger.info(f"Dataset charge : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")

    def replace_sentinels(self):
        before = len(self.df)
        nan_before = self.df.isnull().sum().sum()

        TEXT_SENTINELS = self.config["steps"]["sentinels"]["text_sentinels"]
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].replace(TEXT_SENTINELS, np.nan)

        for rule in self.config["steps"]["sentinels"]["numeric_sentinels"]:
            self.df[rule["column"]] = self.df[rule["column"]].replace(rule["values"], np.nan)

        nan_after = self.df.isnull().sum().sum()
        self.logger.info(f"[Sentinelles] NaN avant={nan_before} -> apres={nan_after} (+{nan_after - nan_before} NaN crees)")
        self.logger.info(f"[Sentinelles] {before} -> {len(self.df)} lignes")

    def remove_duplicates(self):
        before = len(self.df)
        cfg = self.config["steps"]["deduplication"]
        n_exact    = self.df.duplicated().sum()
        n_trade_id = self.df.duplicated(subset='trade_id').sum()
        self.logger.info(f"[Doublons] Doublons exacts={n_exact} | Doublons trade_id={n_trade_id}")
        self.df = self.df.drop_duplicates(subset=cfg["subset"], keep=cfg["keep"])
        self.logger.info(f"[Doublons] {before} -> {len(self.df)} lignes")

    def cast_types(self):
        before = len(self.df)
        cfg = self.config["steps"]["type_casting"]

        for col in cfg["datetime_columns"]:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        for col in cfg["numeric_columns"]:
            nan_avant = self.df[col].isnull().sum()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            nan_apres = self.df[col].isnull().sum()
            if nan_apres > nan_avant:
                self.logger.info(f"[Types] {col} : {nan_apres - nan_avant} valeurs non convertibles -> NaN")

        for col in cfg["categorical_columns"]:
            self.df[col] = self.df[col].astype(str).str.strip().str.lower().replace('nan', np.nan)

        self.logger.info(f"[Types] {before} -> {len(self.df)} lignes")

    def normalize_referentials(self):
        before = len(self.df)
        ASSET_CLASS_MAP = self.config["steps"]["referential_normalization"]["columns"]["asset_class"]["mapping"]
        self.df['asset_class'] = self.df['asset_class'].map(ASSET_CLASS_MAP)
        n_nan_ac = self.df['asset_class'].isnull().sum()
        self.logger.info(f"[asset_class] {n_nan_ac} valeurs non mappees -> NaN | valeurs restantes : {self.df['asset_class'].value_counts().to_dict()}")
        self.logger.info(f"[asset_class] {before} -> {len(self.df)} lignes")

    def fix_financial_coherence(self):
        before = len(self.df)
        rules = {r["id"]: r for r in self.config["steps"]["financial_coherence"]["rules"]}

        # 5a
        r = rules["5a"]
        bad_settle = self.df['settlement_date'] < self.df['trade_date']
        self.df.loc[bad_settle, 'settlement_date'] = self.df.loc[bad_settle, 'trade_date'] + pd.Timedelta(days=r["t_plus_days"])
        self.logger.info(f"[5a settlement] {bad_settle.sum()} lignes : settlement mis a trade_date+{r['t_plus_days']}j (regle T+2 marches actions)")

        # 5b
        bad_bid_ask = self.df['bid'] > self.df['ask']
        self.df.loc[bad_bid_ask, ['bid', 'ask']] = self.df.loc[bad_bid_ask, ['ask', 'bid']].values
        self.logger.info(f"[5b bid/ask] {bad_bid_ask.sum()} lignes : bid et ask permutes (fourchette physiquement impossible inversee)")

        # 5c
        r = rules["5c"]
        mid_calc = (self.df['bid'] + self.df['ask']) / 2
        bad_mid  = (abs(self.df['mid_price'] - mid_calc) / mid_calc.replace(0, np.nan)) > r["tolerance"]
        self.df.loc[bad_mid, 'mid_price'] = mid_calc[bad_mid].round(r["round_decimals"])
        self.logger.info(f"[5c mid_price] {bad_mid.sum()} lignes : mid_price recalcule = (bid+ask)/2 (erreur calcul source Bloomberg)")

        # 5d
        r = rules["5d"]
        bad_price = (self.df['price'] < self.df['bid'] * r["bid_lower_factor"]) | (self.df['price'] > self.df['ask'] * r["ask_upper_factor"])
        self.df.loc[bad_price, 'price'] = self.df.loc[bad_price, 'mid_price']
        self.logger.info(f"[5d price] {bad_price.sum()} lignes : price hors fourchette -> remplace par mid_price (execution hors marche impossible)")

        # 5e
        r = rules["5e"]
        bad_notional = self.df['notional_eur'] < 0
        self.df[r["flag_column"]] = 0
        self.df.loc[bad_notional, r["flag_column"]] = 1
        self.df.loc[bad_notional, 'notional_eur'] = self.df.loc[bad_notional, 'notional_eur'].abs()
        self.logger.info(f"[5e notional] {bad_notional.sum()} lignes : notionnel negatif -> abs() + flag {r['flag_column']}=1 (preserve info position short)")

        # 5f
        r = rules["5f"]
        bad_rating = self.df['credit_rating'].isin(r["investment_grade_ratings"]) & (self.df['default_flag'] == r["default_flag_value"])
        self.df.loc[bad_rating, 'credit_rating'] = r["degraded_rating"]
        self.logger.info(f"[5f rating/defaut] {bad_rating.sum()} lignes : rating degrade a {r['degraded_rating'].upper()} (AAA/AA/A + defaut = contradiction logique impossible)")

        self.logger.info(f"[Incoherences financieres] {before} -> {len(self.df)} lignes")

    def apply_domain_rules(self):
        before = len(self.df)

        for rule in self.config["steps"]["domain_rules"]["rules"]:
            col = rule["column"]
            if "allowed_values" in rule:
                mask = ~self.df[col].isin(rule["allowed_values"])
            elif rule.get("exclusive_min"):
                mask = self.df[col] <= rule["min"]
            else:
                mask = (self.df[col] < rule["min"]) | (self.df[col] > rule["max"])
            self.df.loc[mask, col] = np.nan
            self.logger.info(f"[6 {col}] {mask.sum()} valeurs invalides -> NaN ({rule['reason']})")

        self.logger.info(f"[Regles metier] {before} -> {len(self.df)} lignes")

    def handle_outliers(self):
        # Winsorisation : on borne les valeurs extremes sans les supprimer.
        # En finance, des notionnels ou volatilites extremes sont rares mais reels, les supprimer biaiserait le modele sur les queues de distribution

        before = len(self.df)
        cfg = self.config["steps"]["outliers"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        outlier_cols = cfg["iqr"]["columns"]

        for ax, col in zip(axes, outlier_cols):
            ax.boxplot(self.df[col].dropna(), patch_artist=True, boxprops=dict(facecolor='#90caf9'))
            ax.set_title(f'{col} (avant)', fontsize=9)

        plt.suptitle('Boxplots outliers IQR avant winsorisation', fontweight='bold')
        plt.tight_layout()
        plt.savefig(cfg["boxplot_output"], dpi=120, bbox_inches='tight')
        plt.close()

        for col in outlier_cols:
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr = q3 - q1
            mult = cfg["iqr"]["multiplier"]
            lower, upper = q1 - mult * iqr, q3 + mult * iqr
            n_out = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            self.logger.info(f"[7 IQR] {col}: {n_out} outliers winsories dans [{lower:.2f},{upper:.2f}]")

        iso_cfg      = cfg["isolation_forest"]
        iso_features = iso_cfg["features"]
        iso_df = self.df[iso_features].fillna(self.df[iso_features].median())
        iso = IsolationForest(n_estimators=iso_cfg["n_estimators"], contamination=iso_cfg["contamination"], random_state=iso_cfg["random_state"], n_jobs=-1)
        self.df['is_anomaly_multivariate'] = (iso.fit_predict(iso_df) == -1).astype(int)
        n_anomaly = self.df['is_anomaly_multivariate'].sum()
        self.logger.info(f"[7 IsolationForest] {n_anomaly} anomalies multivariées flagguees (conservées pour revue Risk Officer)")

        self.logger.info(f"[Outliers] {before} -> {len(self.df)} lignes")

    def handle_missing_values(self):
        before = len(self.df)
        cfg = self.config["steps"]["missing_values"]

        # trade_id NaN : un trade sans identifiant est intraitable — on ne peut pas
        # le retrouver dans Murex, l auditer, ni le rattacher a une contrepartie.
        # Suppression obligatoire, pas d imputation possible.
        n_tid_nan = self.df['trade_id'].isnull().sum()
        if n_tid_nan > 0:
            self.df = self.df.dropna(subset=['trade_id'])
            self.logger.info(f"[8 trade_id] {n_tid_nan} lignes supprimées : trade_id NaN = trade intraitable")

        # settlement_date NaT : la regle T+2 est deterministe pour les actions.
        # On calcule trade_date + 2j plutot que d imputer la mediane qui n aurait
        # aucun sens metier ici (le delai de reglement n est pas aleatoire).
        nat_settle = self.df['settlement_date'].isnull()
        if nat_settle.sum() > 0:
            self.df.loc[nat_settle, 'settlement_date'] = self.df.loc[nat_settle, 'trade_date'] + pd.Timedelta(days=2)
            self.logger.info(f"[8 settlement_date] {nat_settle.sum()} NaT -> trade_date+2j (valeur deterministe T+2)")

        # Colonnes numeriques : mediane + flag _was_missing.
        # On choisit la mediane plutot que la moyenne car les distributions financieres
        # sont asymetriques, la moyenne serait tirée par les grosses transactions.
        # Le flag permet au modele de distinguer les vraies valeurs des imputées.
        # Taux de NaN tous inferieurs a 20% apres traitement sentinelles/incoherences,
        # donc on reste dans la regle generale imputer + flag.
        for col in cfg["numeric_imputation"]["columns"]:
            n_nan = self.df[col].isnull().sum()
            rate  = n_nan / len(self.df)
            if n_nan > 0:
                self.df[f'{col}_was_missing'] = self.df[col].isnull().astype(int)
                med = self.df[col].median()
                self.df[col] = self.df[col].fillna(med)
                self.logger.info(f"[8 {col}] {n_nan} NaN ({rate*100:.1f}%) -> mediane={med:.4f} + flag {col}_was_missing")

        # asset_class et sector : mode + flag.
        # La valeur la plus frequente est une imputation acceptable pour ces colonnes
        # car elles ont peu de NaN et une distribution concentree sur quelques valeurs.
        for col in [r["column"] for r in cfg["categorical_imputation"] if r.get("strategy") == "mode"]:
            n_nan = self.df[col].isnull().sum()
            if n_nan > 0:
                self.df[f'{col}_was_missing'] = self.df[col].isnull().astype(int)
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                self.logger.info(f"[8 {col}] {n_nan} NaN -> mode='{mode_val}' + flag")

        # credit_rating : on refuse d imputer le mode (BBB).
        # Une contrepartie sans notation n'est pas une BBB  c'est une entité dont
        # on ne connait pas le risque, ce qui est une information en soi.
        # Imputer BBB introduirait un biais systematique vers le centre de la distribution
        # et masquerait les contreparties les plus risquees du portefeuille.
        # 'nr' (Not Rated) preserve cette distinction semantique pour le modele.
        for rule in [r for r in cfg["categorical_imputation"] if r.get("strategy") == "constant"]:
            col = rule["column"]
            fill_value = rule["fill_value"]
            n_nan = self.df[col].isnull().sum()
            if n_nan > 0:
                self.df[f'{col}_was_missing'] = self.df[col].isnull().astype(int)
                self.df[col] = self.df[col].fillna(fill_value)
                self.logger.info(f"[8 {col}] {n_nan} NaN -> '{fill_value}' (Not Rated) : imputer le mode BBB serait trompeur pour un modele de risque)")

        self.logger.info(f"[Valeurs manquantes] {before} -> {len(self.df)} lignes")

    def pseudonymize(self):
        # Les colonnes :
        # counterparty_name et trader_id contiennent des informations d'identification directe (RGPD Art.4(1))
        # qui permettent d'identifier une personne physique ou morale.
        # Ces données sont sensibles car elles peuvent être utilisées pour retracer les transactions à des entités spécifiques, c
        # ce qui pose un risque de confidentialité.
        #  La pseudonymisation par hashage irreversible permet de protéger ces données
        # tout en conservant la possibilité d'analyse au niveau agrégé sans exposer les identités réelles.

        before = len(self.df)
        cfg = self.config["steps"]["pseudonymization"]
        salt = os.environ.get(cfg["salt_env_var"], cfg["salt_default"])

        for rule in cfg["columns"]:
            col = rule["column"]
            self.df[f'{col}_hash'] = self.df[col].astype(str).apply(
                lambda x: hashlib.sha256(f"{salt}{x}".encode()).hexdigest()
            )
            self.df = self.df.drop(columns=[col])
            self.logger.info(f"[9 PII] '{col}' pseudonymise -> '{col}_hash' (SHA-256+salt). RGPD Art.4(5) : donnée identifiante indirecte)")

        self.logger.info(f"[Pseudonymisation RGPD] {before} -> {len(self.df)} lignes")

    def quality_report(self):
        cfg = self.config["steps"]["quality_report"]
        weights = cfg["dqs_weights"]

        nan_total   = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        completude  = 1 - nan_total / total_cells
        unicite     = 1 - self.df.duplicated(subset='trade_id').mean()
        dqs         = (completude * weights["completeness"] + unicite * weights["uniqueness"]) * 100

        raw_completude = (1 - self.df_raw.isnull().mean().mean()) * 100

        print("\n" + "=" * 60)
        print("RAPPORT QUALITE AVANT / APRES")
        print("=" * 60)
        print(f"{'Metrique':<35} {'Avant':>10} {'Apres':>10}")
        print("-" * 60)
        print(f"{'Nb lignes':<35} {self.df_raw.shape[0]:>10} {self.df.shape[0]:>10}")
        print(f"{'Nb colonnes':<35} {self.df_raw.shape[1]:>10} {self.df.shape[1]:>10}")
        print(f"{'Completude globale':<35} {raw_completude:>9.2f}% {completude*100:>9.2f}%")
        print(f"{'Doublons trade_id':<35} {self.df_raw.duplicated(subset='trade_id').sum():>10} {self.df.duplicated(subset='trade_id').sum():>10}")
        print(f"{'Data Quality Score (DQS)':<35} {'N/A':>10} {dqs:>9.2f}%")
        print("=" * 60)

    def save(self):
        output_path = self.config["io"]["output_path"]
        self.df.to_csv(output_path, index=False)
        self.logger.info(f"Dataset nettoyé et sauvegardé : {output_path}")
        print(f"Shape finale : {self.df.shape}")
