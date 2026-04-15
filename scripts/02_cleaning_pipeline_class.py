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

from classes.DataCleaner import DataCleaner

cleaner = DataCleaner("config/cleaning.json")

# Configuration du logging (ne pas modifier)
cleaner.setup_logging()

# ============================================================
# CHARGEMENT (ne pas modifier)
# ============================================================
cleaner.load()

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
cleaner.replace_sentinels()

# ============================================================
# ETAPE 2 — Suppression des doublons
# ============================================================
# CONSIGNE :
# Supprimez les doublons sur la cle metier trade_id.
# Justifiez dans un commentaire : pourquoi garder "first" ou "last" ?
# Dans le contexte Murex, quel enregistrement est le plus fiable ?
#
# Loggez : nb de doublons exacts, nb de doublons sur trade_id, shape finale.
cleaner.remove_duplicates()

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
cleaner.cast_types()

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
cleaner.normalize_referentials()

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
cleaner.fix_financial_coherence()

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
cleaner.apply_domain_rules()

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
cleaner.handle_outliers()

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
cleaner.handle_missing_values()

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
cleaner.pseudonymize()

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
cleaner.quality_report()
cleaner.save()
