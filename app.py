# ============================================================
# APPLICATION STREAMLIT
# PREDICTION DU RISQUE DE PANNE DES GAB
# OMOA - DATA SCIENCE
# ============================================================

from pathlib import Path
import io
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

st.set_page_config(
    page_title="Prédiction du risque de panne des GAB",
    page_icon="🏧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CHEMINS DES FICHIERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOSSIER_MODELES = BASE_DIR / "production_omoa_models"

CHEMIN_PIPELINE = (
    DOSSIER_MODELES / "pipeline_lgb_gab.joblib"
)

CHEMIN_VARIABLES = (
    DOSSIER_MODELES / "noms_variables_explicatives.joblib"
)

CHEMIN_IMAGE_FOND = BASE_DIR / "image_gab.jpg"


# ============================================================
# VARIABLES UTILISEES PAR LE MODELE
# ============================================================

VARIABLES_MODELE = [
    "Type_GAB",
    "Marque",
    "Anciennete_GAB_annees",
    "Temps_depuis_derniere_maintenance_preventive_jours",
    "Temps_moyen_entre_pannes",
    "Nombre_maintenances_preventives",
    "Temps_moyen_reparation",
    "Temps_depuis_derniere_panne_jours"
]

VARIABLES_NUMERIQUES = [
    "Anciennete_GAB_annees",
    "Temps_depuis_derniere_maintenance_preventive_jours",
    "Temps_moyen_entre_pannes",
    "Nombre_maintenances_preventives",
    "Temps_moyen_reparation",
    "Temps_depuis_derniere_panne_jours"
]

VARIABLES_CATEGORIELLES = [
    "Type_GAB",
    "Marque"
]


# ============================================================
# MODALITES DE SECOURS
# ============================================================

MODALITES_MARQUE = [
    "NCR",
    "WINCOR"
]

MODALITES_TYPE_GAB = [
    "DD - Dual Dispenser",
    "Dépôt Cash + Dépôt Chèque",
    "Dépôt cash",
    "MF - Multifonction",
    "SS - Standard"
]


# ============================================================
# CORRESPONDANCE DES RISQUES
# ============================================================

NIVEAUX_RISQUE = {
    0: "Faible",
    1: "Moyen",
    2: "Élevé"
}

ORDRE_RISQUE = [
    "Faible",
    "Moyen",
    "Élevé"
]

ICONES_RISQUE = {
    "Faible": "🟢",
    "Moyen": "🟠",
    "Élevé": "🔴"
}


# ============================================================
# COLONNES ATTENDUES POUR L'IMPORTATION
# ============================================================

COLONNES_IMPORTATION = [
    "GAB_ID",
    "Client",
    "Anciennete_GAB_annees",
    "Nombre_maintenances_preventives",
    "Temps_depuis_derniere_maintenance_preventive_jours",
    "Temps_depuis_derniere_panne_jours",
    "Temps_moyen_entre_pannes",
    "Temps_moyen_reparation",
    "Marque",
    "Type_GAB"
]


def generer_recommandation(
    ligne,
    risque,
    probabilites_par_classe
):
    """
    Génère une recommandation uniquement à partir
    du niveau de risque prédit par le modèle.
    """

    probabilite_risque_eleve = float(
        probabilites_par_classe.get(
            2,
            0.0
        )
    )

    if risque == "Élevé":

        priorite = "Priorité élevée"

        recommandations = [
            (
                "Programmer une vérification technique "
                "prioritaire du GAB."
            ),
            (
                "Réaliser un diagnostic approfondi afin "
                "d'identifier les causes potentielles "
                "de défaillance."
            ),
            (
                "Analyser l'historique des pannes et des "
                "interventions avant toute nouvelle opération."
            )
        ]

    elif risque == "Moyen":

        priorite = "Priorité moyenne"

        recommandations = [
            (
                "Renforcer la surveillance du GAB."
            ),
            (
                "Programmer un contrôle préventif selon "
                "le planning de maintenance."
            ),
            (
                "Suivre l'évolution du niveau de risque lors "
                "des prochaines mises à jour des données."
            )
        ]

    else:

        priorite = "Priorité normale"

        recommandations = [
            (
                "Maintenir le calendrier habituel "
                "de maintenance préventive."
            ),
            (
                "Poursuivre la surveillance régulière "
                "du fonctionnement du GAB."
            ),
            (
                "Actualiser la prédiction lors de la prochaine "
                "mise à jour des données."
            )
        ]

    return {
        "priorite":
            priorite,

        "probabilite_risque_eleve":
            probabilite_risque_eleve,

        "recommandations":
            recommandations
    }

# ============================================================
# IMAGE D'ARRIERE-PLAN
# ============================================================

def ajouter_image_arriere_plan(chemin_image):
    """Ajoute une image d'arrière-plan à l'application."""

    chemin_image = Path(chemin_image)

    if not chemin_image.exists():
        return

    with open(chemin_image, "rb") as fichier_image:
        image_encodee = base64.b64encode(
            fichier_image.read()
        ).decode("utf-8")

    extension = chemin_image.suffix.lower().replace(".", "")

    if extension == "jpg":
        extension = "jpeg"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(255, 255, 255, 0.90),
                    rgba(255, 255, 255, 0.90)
                ),
                url("data:image/{extension};base64,{image_encodee}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(248, 249, 250, 0.96);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


ajouter_image_arriere_plan(CHEMIN_IMAGE_FOND)


# ============================================================
# STYLE CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        border: 1px solid #dddddd;
    }

    .risk-value {
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        border: 1px solid #dddddd;
        text-align: center;
    }

    .section-title {
        font-size: 23px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .small-text {
        font-size: 14px;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHARGEMENT SECURISE DU PIPELINE
# ============================================================

@st.cache_resource
def charger_modele(chemin_pipeline, date_modification):
    """Charge et vérifie le pipeline complet."""

    chemin_pipeline = Path(chemin_pipeline)

    if not chemin_pipeline.exists():
        raise FileNotFoundError(
            f"Pipeline introuvable : {chemin_pipeline.resolve()}"
        )

    pipeline = joblib.load(chemin_pipeline)

    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "Le fichier chargé ne contient pas un pipeline valide."
        )

    etapes_attendues = {"preprocesseur", "modele"}
    etapes_disponibles = set(pipeline.named_steps.keys())

    if not etapes_attendues.issubset(etapes_disponibles):
        raise ValueError(
            "Le pipeline doit contenir les étapes "
            "'preprocesseur' et 'modele'."
        )

    modele = pipeline.named_steps["modele"]

    if not hasattr(modele, "classes_"):
        raise ValueError(
            "Le modèle chargé ne possède pas l'attribut classes_."
        )

    return pipeline


@st.cache_data
def charger_variables(chemin_variables, date_modification):
    """Charge la liste ordonnée des variables explicatives."""

    chemin_variables = Path(chemin_variables)

    if not chemin_variables.exists():
        raise FileNotFoundError(
            f"Fichier des variables introuvable : "
            f"{chemin_variables.resolve()}"
        )

    return list(joblib.load(chemin_variables))


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

try:
    if not CHEMIN_PIPELINE.exists():
        raise FileNotFoundError(
            f"Pipeline introuvable : {CHEMIN_PIPELINE.resolve()}"
        )

    if not CHEMIN_VARIABLES.exists():
        raise FileNotFoundError(
            f"Variables introuvables : {CHEMIN_VARIABLES.resolve()}"
        )

    date_modification_pipeline = CHEMIN_PIPELINE.stat().st_mtime
    date_modification_variables = CHEMIN_VARIABLES.stat().st_mtime

    modele_lgb = charger_modele(
        str(CHEMIN_PIPELINE),
        date_modification_pipeline
    )

    variables_sauvegardees = charger_variables(
        str(CHEMIN_VARIABLES),
        date_modification_variables
    )

    preprocesseur_final = modele_lgb.named_steps["preprocesseur"]
    modele_final = modele_lgb.named_steps["modele"]
    classes_modele = list(modele_final.classes_)

except Exception as erreur:
    st.error("Impossible de charger le modèle de prédiction.")
    st.exception(erreur)
    st.stop()


# ============================================================
# VERIFICATION DU MODELE
# ============================================================

if variables_sauvegardees != VARIABLES_MODELE:
    st.error(
        "Les variables enregistrées avec le modèle ne "
        "correspondent pas à celles de l'application."
    )

    st.write("Variables enregistrées :", variables_sauvegardees)
    st.write("Variables de l'application :", VARIABLES_MODELE)
    st.stop()


classes_chargees = {
    int(classe)
    for classe in classes_modele
}

if classes_chargees != {0, 1, 2}:
    st.error(
        "Les classes du modèle ne correspondent pas "
        "aux niveaux de risque attendus."
    )

    st.write("Classes chargées :", classes_modele)
    st.stop()


# ============================================================
# MODALITES APPRISES PAR LE PIPELINE
# ============================================================

try:
    encodeur_categoriel = (
        preprocesseur_final
        .named_transformers_["variables_categorielles"]
    )

    categories_apprises = encodeur_categoriel.categories_

    # Ordre utilisé pendant l'entraînement :
    # Type_GAB, puis Marque
    MODALITES_TYPE_GAB = [
        str(valeur)
        for valeur in categories_apprises[0]
    ]

    MODALITES_MARQUE = [
        str(valeur)
        for valeur in categories_apprises[1]
    ]

except Exception:
    pass


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def convertir_classe(classe):
    """Convertit une classe NumPy en entier Python."""

    try:
        return int(classe)
    except (TypeError, ValueError):
        return int(float(classe))


def obtenir_risque_depuis_classe(classe):
    """Retourne le niveau de risque associé à une classe."""

    classe = convertir_classe(classe)

    return NIVEAUX_RISQUE.get(
        classe,
        "Classe inconnue"
    )


def preparer_entrees_modele(df):
    """Prépare les données brutes destinées au pipeline."""

    colonnes_manquantes = [
        variable
        for variable in VARIABLES_MODELE
        if variable not in df.columns
    ]

    if colonnes_manquantes:
        raise ValueError(
            "Variables manquantes pour la prédiction : "
            + ", ".join(colonnes_manquantes)
        )

    X = df.loc[:, VARIABLES_MODELE].copy()

    for variable in VARIABLES_CATEGORIELLES:
        if X[variable].isna().any():
            raise ValueError(
                f"La variable '{variable}' contient "
                "des valeurs manquantes."
            )

        X[variable] = (
            X[variable]
            .astype("string")
            .str.strip()
        )

    for variable in VARIABLES_NUMERIQUES:
        X[variable] = pd.to_numeric(
            X[variable],
            errors="coerce"
        )

        if X[variable].isna().any():
            raise ValueError(
                f"La variable '{variable}' contient "
                "des valeurs manquantes ou invalides."
            )

        if not np.isfinite(
            X[variable].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"La variable '{variable}' contient "
                "des valeurs infinies."
            )

    return X


def verifier_valeurs_numeriques(df):
    """Vérifie la validité des variables numériques."""

    erreurs = []

    for variable in VARIABLES_NUMERIQUES:
        if variable not in df.columns:
            erreurs.append(
                f"Colonne absente : {variable}"
            )
            continue

        valeurs = pd.to_numeric(
            df[variable],
            errors="coerce"
        )

        if valeurs.isna().any():
            erreurs.append(
                f"Valeurs invalides ou manquantes dans : {variable}"
            )
            continue

        if not np.isfinite(
            valeurs.to_numpy(dtype=float)
        ).all():
            erreurs.append(
                f"Valeurs infinies détectées dans : {variable}"
            )

    return erreurs


def verifier_modalites_importees(df):
    """Vérifie les modalités catégorielles importées."""

    erreurs = []

    if "Marque" in df.columns:
        marques = (
            df["Marque"]
            .dropna()
            .astype("string")
            .str.strip()
            .unique()
        )

        marques_inconnues = [
            str(valeur)
            for valeur in marques
            if valeur not in MODALITES_MARQUE
        ]

        if marques_inconnues:
            erreurs.append(
                "Marque(s) inconnue(s) : "
                + ", ".join(marques_inconnues)
            )

    if "Type_GAB" in df.columns:
        types_gab = (
            df["Type_GAB"]
            .dropna()
            .astype("string")
            .str.strip()
            .unique()
        )

        types_inconnus = [
            str(valeur)
            for valeur in types_gab
            if valeur not in MODALITES_TYPE_GAB
        ]

        if types_inconnus:
            erreurs.append(
                "Type(s) de GAB inconnu(s) : "
                + ", ".join(types_inconnus)
            )

    return erreurs


def verifier_donnees_importees(df):
    """Vérifie la structure et le contenu des données importées."""

    erreurs = []

    colonnes_manquantes = [
        colonne
        for colonne in COLONNES_IMPORTATION
        if colonne not in df.columns
    ]

    if colonnes_manquantes:
        erreurs.append(
            "Colonnes manquantes : "
            + ", ".join(colonnes_manquantes)
        )
        return erreurs

    erreurs.extend(
        verifier_modalites_importees(df)
    )

    erreurs.extend(
        verifier_valeurs_numeriques(df)
    )

    return erreurs


def lire_fichier_importe(fichier):
    """Lit un fichier CSV, XLSX ou XLS."""

    nom_fichier = fichier.name.lower()

    if nom_fichier.endswith(".csv"):
        try:
            dataframe = pd.read_csv(
                fichier,
                sep=None,
                engine="python",
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            fichier.seek(0)

            dataframe = pd.read_csv(
                fichier,
                sep=None,
                engine="python",
                encoding="latin-1"
            )

    elif nom_fichier.endswith(".xlsx"):
        dataframe = pd.read_excel(
            fichier,
            engine="openpyxl"
        )

    elif nom_fichier.endswith(".xls"):
        dataframe = pd.read_excel(
            fichier,
            engine="xlrd"
        )

    else:
        raise ValueError(
            "Format non pris en charge. "
            "Formats acceptés : CSV, XLSX et XLS."
        )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    return dataframe


def construire_ligne_diagnostic(
    type_gab_selectionne,
    marque_selectionnee,
    anciennete,
    temps_depuis_derniere_maintenance,
    temps_moyen_entre_pannes,
    nombre_maintenances,
    temps_moyen_reparation,
    temps_depuis_derniere_panne
):
    """Construit une observation brute."""

    valeurs = {
        "Type_GAB":
            type_gab_selectionne,

        "Marque":
            marque_selectionnee,

        "Anciennete_GAB_annees":
            anciennete,

        "Temps_depuis_derniere_maintenance_preventive_jours":
            temps_depuis_derniere_maintenance,

        "Temps_moyen_entre_pannes":
            temps_moyen_entre_pannes,

        "Nombre_maintenances_preventives":
            nombre_maintenances,

        "Temps_moyen_reparation":
            temps_moyen_reparation,

        "Temps_depuis_derniere_panne_jours":
            temps_depuis_derniere_panne
    }

    return pd.DataFrame(
        [valeurs],
        columns=VARIABLES_MODELE
    )


def executer_prediction(df):
    """Effectue une prédiction individuelle."""

    X = preparer_entrees_modele(df)

    prediction = np.asarray(
        modele_lgb.predict(X)
    ).ravel()

    probabilites = np.asarray(
        modele_lgb.predict_proba(X)
    )

    if len(prediction) != 1:
        raise ValueError(
            "Le diagnostic individuel attend une seule observation."
        )

    if probabilites.shape != (
        1,
        len(classes_modele)
    ):
        raise ValueError(
            "La matrice des probabilités est incorrecte."
        )

    classe_predite = convertir_classe(
        prediction[0]
    )

    risque = obtenir_risque_depuis_classe(
        classe_predite
    )

    probabilites_par_classe = {}

    for indice, classe in enumerate(classes_modele):
        classe = convertir_classe(classe)

        probabilites_par_classe[classe] = float(
            probabilites[0, indice]
        )

    return (
        classe_predite,
        risque,
        probabilites_par_classe
    )


def executer_predictions_multiples(df):
    """Effectue les prédictions pour plusieurs GAB."""

    X = preparer_entrees_modele(df)

    predictions = np.asarray(
        modele_lgb.predict(X)
    ).ravel()

    probabilites = np.asarray(
        modele_lgb.predict_proba(X)
    )

    if len(predictions) != len(X):
        raise ValueError(
            "Le nombre de prédictions ne correspond pas "
            "au nombre d'observations."
        )

    return predictions, probabilites


# ============================================================
# RECOMMANDATIONS LIEES AUX PREDICTIONS
# ============================================================

def generer_recommandation(
    ligne,
    risque,
    probabilites_par_classe
):
    """
    Génère une recommandation uniquement à partir
    du niveau de risque prédit par le modèle.
    """

    probabilite_risque_eleve = float(
        probabilites_par_classe.get(
            2,
            0.0
        )
    )

    if risque == "Élevé":

        priorite = "Priorité élevée"

        recommandations = [
            (
                "Programmer une vérification technique "
                "prioritaire du GAB."
            ),
            (
                "Réaliser un diagnostic approfondi afin "
                "d'identifier les causes potentielles "
                "de défaillance."
            ),
            (
                "Analyser l'historique des pannes et des "
                "interventions avant toute nouvelle opération."
            )
        ]

    elif risque == "Moyen":

        priorite = "Priorité moyenne"

        recommandations = [
            (
                "Renforcer la surveillance du GAB."
            ),
            (
                "Programmer un contrôle préventif selon "
                "le planning de maintenance."
            ),
            (
                "Suivre l'évolution du niveau de risque lors "
                "des prochaines mises à jour des données."
            )
        ]

    else:

        priorite = "Priorité normale"

        recommandations = [
            (
                "Maintenir le calendrier habituel "
                "de maintenance préventive."
            ),
            (
                "Poursuivre la surveillance régulière "
                "du fonctionnement du GAB."
            ),
            (
                "Actualiser la prédiction lors de la prochaine "
                "mise à jour des données."
            )
        ]

    return {
        "priorite":
            priorite,

        "probabilite_risque_eleve":
            probabilite_risque_eleve,

        "recommandations":
            recommandations
    }

def generer_action_synthetique(
    ligne,
    risque,
    probabilite_elevee
):
    """
    Génère une action synthétique uniquement à partir
    du niveau de risque prédit.
    """

    if risque == "Élevé":

        return (
            "Vérification technique prioritaire "
            "et diagnostic approfondi"
        )

    if risque == "Moyen":

        return (
            "Surveillance renforcée et contrôle préventif"
        )

    return (
        "Maintien du suivi et du calendrier "
        "habituel de maintenance"
    )


def afficher_resultat_prediction(
    classe_predite,
    risque,
    probabilites_par_classe
):
    """
    Affiche le niveau de risque prédit
    et les probabilités associées aux trois classes.
    """

    icone = ICONES_RISQUE.get(
        risque,
        "⚪"
    )

    # ========================================================
    # NIVEAU DE RISQUE PREDIT
    # ========================================================

    with st.container(border=True):

        st.markdown(
            "### Niveau de risque prédit"
        )

        if risque == "Élevé":

            st.error(
                f"🔴 Risque {risque}"
            )

        elif risque == "Moyen":

            st.warning(
                f"🟠 Risque {risque}"
            )

        elif risque == "Faible":

            st.success(
                f"🟢 Risque {risque}"
            )

        else:

            st.info(
                f"{icone} {risque}"
            )

    # ========================================================
    # PROBABILITES PREDITES
    # ========================================================

    st.markdown(
        "### Probabilités prédites"
    )

    col1, col2, col3 = st.columns(3)

    affichages_classes = [
        (col1, 0, "🟢"),
        (col2, 1, "🟠"),
        (col3, 2, "🔴")
    ]

    for colonne, classe, icone_classe in affichages_classes:

        probabilite = float(
            probabilites_par_classe.get(
                classe,
                0.0
            )
        )

        probabilite = min(
            max(
                probabilite,
                0.0
            ),
            1.0
        )

        with colonne:

            with st.container(border=True):

                st.metric(
                    label=(
                        f"{icone_classe} "
                        f"Risque {NIVEAUX_RISQUE[classe]}"
                    ),
                    value=(
                        f"{probabilite * 100:.2f} %"
                    )
                )

                st.progress(
                    probabilite
                )

def afficher_recommandation(recommandation):
    """Affiche une recommandation opérationnelle."""

    st.markdown(
        "### 🛠️ Recommandation opérationnelle"
    )

    priorite = recommandation["priorite"]

    if priorite == "Priorité élevée":
        st.error(priorite)

    elif priorite == "Priorité moyenne":
        st.warning(priorite)

    else:
        st.success(priorite)

    st.write(
        "Probabilité associée au risque élevé : "
        f"**{recommandation['probabilite_risque_eleve'] * 100:.2f} %**"
    )

    for numero, action in enumerate(
        recommandation["recommandations"],
        start=1
    ):
        st.write(
            f"{numero}. {action}"
        )

    st.caption(
        "Ces recommandations constituent une aide à la décision. "
        "Les seuils opérationnels doivent être validés par les "
        "responsables et techniciens de maintenance."
    )


def convertir_dataframe_excel(dataframe):
    """Convertit un DataFrame en fichier Excel en mémoire."""

    sortie = io.BytesIO()

    with pd.ExcelWriter(
        sortie,
        engine="openpyxl"
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Predictions_GAB"
        )

    sortie.seek(0)

    return sortie.getvalue()


# ============================================================
# INTERFACE LATERALE
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px;
        ">
            <h2>🏧 OMOA</h2>
            <p>Data Science</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "📥 Importation des données",
            "🔍 Diagnostic individuel",
            "👥 Analyse par client",
            "📊 Analyse des risques",
            "👩‍💻 Auteur"
        ]
    )

    st.markdown("---")

    st.markdown(
        "### ⚙️ Informations techniques"
    )

    st.write(
        f"Variables d'entrée : "
        f"**{len(VARIABLES_MODELE)}**"
    )

    if variables_sauvegardees:
        st.write(
            "Variables explicatives : "
            f"**{len(variables_sauvegardees)}**"
        )

    st.write("Modèle : **LightGBM**")
    st.write("Classes : **3 niveaux de risque**")

    if st.button(
        "🔄 Recharger le modèle",
        use_container_width=True
    ):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


# ============================================================
# TITRE PRINCIPAL
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🏧 Prédiction du risque de panne des GAB
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Application de modélisation prédictive développée
        dans le cadre du mémoire de Master 2 Data Science
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABLEAU DE BORD
# ============================================================

if page == "🏠 Accueil":
    st.markdown(
        """
        <div class="section-title">
            Présentation de l'application
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Cette application permet d'estimer le niveau de risque
        de panne d'un Guichet Automatique Bancaire à partir
        de caractéristiques techniques et historiques de
        maintenance.
        """
    )

    st.info(
        """
        Le modèle distingue trois niveaux de risque :
        **Faible**, **Moyen** et **Élevé**.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🔍</h3>
                <b>Diagnostic individuel</b>
                <p>Évaluer le risque d'un GAB</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>📥</h3>
                <b>Importation</b>
                <p>Prédire le risque de plusieurs GAB</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>📊</h3>
                <b>Analyse</b>
                <p>Explorer les niveaux de risque</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🔬 Variables utilisées")

    tableau_variables = pd.DataFrame({
        "Variable": VARIABLES_MODELE,
        "Type": [
            "Qualitative",
            "Qualitative",
            "Quantitative",
            "Quantitative",
            "Quantitative",
            "Quantitative",
            "Quantitative",
            "Quantitative"
        ]
    })

    st.dataframe(
        tableau_variables,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# IMPORTATION DES DONNEES
# ============================================================

elif page == "📥 Importation des données":
    st.markdown(
        """
        <div class="section-title">
            📥 Importation des données
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Importez un fichier CSV ou Excel contenant
        les caractéristiques des GAB à analyser.
        """
    )

    st.info(
        """
        Formats acceptés : **CSV**, **XLSX** et **XLS**.

        Les colonnes **Marque** et **Type_GAB** doivent contenir
        leurs modalités textuelles.
        """
    )

    fichier = st.file_uploader(
        "Choisir un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"]
    )

    if fichier is not None:
        try:
            df_import = lire_fichier_importe(
                fichier
            )

            st.session_state[
                "donnees_importees"
            ] = df_import.copy()

            st.success(
                f"{len(df_import)} ligne(s) importée(s)."
            )

            st.markdown(
                "### Aperçu des données"
            )

            st.dataframe(
                df_import.head(20),
                use_container_width=True,
                hide_index=True
            )

            erreurs = verifier_donnees_importees(
                df_import
            )

            if erreurs:
                st.error(
                    "Des problèmes ont été détectés :"
                )

                for erreur in erreurs:
                    st.warning(erreur)

            else:
                st.success(
                    "Les données sont compatibles avec le modèle."
                )

                st.markdown(
                    "### Variables transmises au modèle"
                )

                X_import = preparer_entrees_modele(
                    df_import
                )

                st.dataframe(
                    X_import.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                if st.button(
                    "🚀 Lancer les prédictions",
                    type="primary"
                ):
                    try:
                        predictions, probabilites = (
                            executer_predictions_multiples(
                                df_import
                            )
                        )

                        resultats = df_import.copy()

                        resultats[
                            "Classe_Risque_PREDITE"
                        ] = [
                            obtenir_risque_depuis_classe(
                                classe
                            )
                            for classe in predictions
                        ]

                        for indice, classe in enumerate(
                            classes_modele
                        ):
                            classe_entiere = convertir_classe(
                                classe
                            )

                            nom_risque = NIVEAUX_RISQUE[
                                classe_entiere
                            ]

                            resultats[
                                f"Prob_Risque_{nom_risque}"
                            ] = np.round(
                                probabilites[:, indice],
                                6
                            )

                        resultats[
                            "Priorite_Intervention"
                        ] = resultats[
                            "Classe_Risque_PREDITE"
                        ].map({
                            "Faible": "Normale",
                            "Moyen": "Moyenne",
                            "Élevé": "Élevée"
                        })

                        resultats[
                            "Action_Recommandee"
                        ] = [
                            generer_action_synthetique(
                                ligne=resultats.iloc[indice],
                                risque=resultats.iloc[indice][
                                    "Classe_Risque_PREDITE"
                                ],
                                probabilite_elevee=float(
                                    resultats.iloc[indice][
                                        "Prob_Risque_Élevé"
                                    ]
                                )
                            )
                            for indice in range(len(resultats))
                        ]

                        st.session_state[
                            "resultats_predictions"
                        ] = resultats.copy()

                        st.success(
                            "Prédictions réalisées avec succès."
                        )

                        st.markdown("### Résultats")

                        st.dataframe(
                            resultats,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.markdown(
                            "### Répartition des niveaux de risque"
                        )

                        repartition = (
                            resultats[
                                "Classe_Risque_PREDITE"
                            ]
                            .value_counts()
                            .reindex(
                                ORDRE_RISQUE,
                                fill_value=0
                            )
                        )

                        st.bar_chart(repartition)

                        csv_resultats = resultats.to_csv(
                            index=False,
                            sep=";"
                        ).encode("utf-8-sig")

                        fichier_excel = convertir_dataframe_excel(
                            resultats
                        )

                        col_csv, col_excel = st.columns(2)

                        with col_csv:
                            st.download_button(
                                label="⬇️ Télécharger en CSV",
                                data=csv_resultats,
                                file_name=(
                                    "resultats_prediction_gab.csv"
                                ),
                                mime="text/csv",
                                use_container_width=True
                            )

                        with col_excel:
                            st.download_button(
                                label="⬇️ Télécharger en Excel",
                                data=fichier_excel,
                                file_name=(
                                    "resultats_prediction_gab.xlsx"
                                ),
                                mime=(
                                    "application/vnd.openxmlformats-"
                                    "officedocument.spreadsheetml.sheet"
                                ),
                                use_container_width=True
                            )

                    except Exception as erreur:
                        st.error(
                            "Une erreur est survenue pendant "
                            "la prédiction."
                        )
                        st.exception(erreur)

        except Exception as erreur:
            st.error(
                "Impossible de lire le fichier importé."
            )
            st.exception(erreur)


# ============================================================
# DIAGNOSTIC INDIVIDUEL
# ============================================================

elif page == "🔍 Diagnostic individuel":
    st.markdown(
        """
        <div class="section-title">
            🔍 Diagnostic individuel d'un GAB
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Renseignez les caractéristiques du GAB afin d'obtenir
        une estimation de son niveau de risque de panne.
        """
    )

    st.info(
        """
        La recommandation est générée à partir de la prédiction
        et des caractéristiques renseignées.
        """
    )

    st.markdown(
        "### 1. Caractéristiques du GAB"
    )

    col1, col2 = st.columns(2)

    with col1:
        marque = st.selectbox(
            "Marque",
            MODALITES_MARQUE
        )

    with col2:
        type_gab = st.selectbox(
            "Type de GAB",
            MODALITES_TYPE_GAB
        )

    st.markdown(
        "### 2. Informations techniques et maintenance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        anciennete = st.number_input(
            "Ancienneté du GAB (années)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

    with col2:
        nombre_maintenances = st.number_input(
            "Nombre de maintenances préventives",
            min_value=0,
            value=2,
            step=1
        )

    with col3:
        temps_depuis_derniere_maintenance = st.number_input(
            "Temps depuis dernière maintenance préventive (jours)",
            min_value=0.0,
            value=90.0,
            step=1.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        temps_moyen_entre_pannes = st.number_input(
            "Temps moyen entre pannes",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    with col2:
        temps_moyen_reparation = st.number_input(
            "Temps moyen de réparation",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

    with col3:
        temps_depuis_derniere_panne = st.number_input(
            "Temps depuis la dernière panne (jours)",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    st.markdown("---")

    if st.button(
        "🔮 Prédire le niveau de risque",
        type="primary",
        use_container_width=True
    ):
        try:
            ligne = construire_ligne_diagnostic(
                type_gab_selectionne=type_gab,
                marque_selectionnee=marque,
                anciennete=anciennete,
                temps_depuis_derniere_maintenance=(
                    temps_depuis_derniere_maintenance
                ),
                temps_moyen_entre_pannes=(
                    temps_moyen_entre_pannes
                ),
                nombre_maintenances=nombre_maintenances,
                temps_moyen_reparation=temps_moyen_reparation,
                temps_depuis_derniere_panne=(
                    temps_depuis_derniere_panne
                )
            )

            classe_predite, risque, probabilites = (
                executer_prediction(ligne)
            )

            recommandation = generer_recommandation(
                ligne=ligne,
                risque=risque,
                probabilites_par_classe=probabilites
            )

            st.session_state["resultat_individuel"] = (
                ligne.copy()
            )

            st.session_state["prediction_individuelle"] = (
                classe_predite
            )

            st.session_state["risque_individuel"] = risque

            st.session_state["probabilites_individuelles"] = (
                probabilites
            )

            st.session_state[
                "recommandation_individuelle"
            ] = recommandation

            afficher_resultat_prediction(
                classe_predite,
                risque,
                probabilites
            )

            afficher_recommandation(
                recommandation
            )

            with st.expander(
                "🔎 Voir les variables envoyées au modèle"
            ):
                st.dataframe(
                    ligne,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as erreur:
            st.error(
                "Impossible d'effectuer la prédiction."
            )
            st.exception(erreur)

    elif "prediction_individuelle" in st.session_state:
        afficher_resultat_prediction(
            st.session_state["prediction_individuelle"],
            st.session_state["risque_individuel"],
            st.session_state["probabilites_individuelles"]
        )

        if "recommandation_individuelle" in st.session_state:
            afficher_recommandation(
                st.session_state[
                    "recommandation_individuelle"
                ]
            )


# ============================================================
# ANALYSE PAR CLIENT
# ============================================================

elif page == "👥 Analyse par client":
    st.markdown(
        """
        <div class="section-title">
            👥 Analyse des risques par client
        </div>
        """,
        unsafe_allow_html=True
    )

    if "resultats_predictions" not in st.session_state:
        st.info(
            """
            Importez d'abord les données et lancez les
            prédictions dans l'onglet « Importation des données ».
            """
        )

    else:
        resultats = st.session_state[
            "resultats_predictions"
        ]

        if "Client" not in resultats.columns:
            st.warning(
                "La colonne Client n'est pas disponible."
            )

        else:
            clients = sorted(
                resultats["Client"]
                .dropna()
                .astype(str)
                .unique()
            )

            if not clients:
                st.warning(
                    "Aucun client disponible."
                )

            else:
                client_selectionne = st.selectbox(
                    "Sélectionner un client",
                    clients
                )

                donnees_client = resultats[
                    resultats["Client"].astype(str)
                    == client_selectionne
                ]

                st.markdown(
                    f"### Client : {client_selectionne}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Nombre de GAB",
                        len(donnees_client)
                    )

                with col2:
                    nombre_eleve = (
                        donnees_client[
                            "Classe_Risque_PREDITE"
                        ]
                        .eq("Élevé")
                        .sum()
                    )

                    st.metric(
                        "GAB à risque élevé",
                        int(nombre_eleve)
                    )

                with col3:
                    nombre_moyen = (
                        donnees_client[
                            "Classe_Risque_PREDITE"
                        ]
                        .eq("Moyen")
                        .sum()
                    )

                    st.metric(
                        "GAB à risque moyen",
                        int(nombre_moyen)
                    )

                st.markdown(
                    "### Répartition des risques"
                )

                repartition = (
                    donnees_client[
                        "Classe_Risque_PREDITE"
                    ]
                    .value_counts()
                    .reindex(
                        ORDRE_RISQUE,
                        fill_value=0
                    )
                )

                st.bar_chart(repartition)

                st.markdown(
                    "### Détail des GAB"
                )

                st.dataframe(
                    donnees_client,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# ANALYSE DES RISQUES
# ============================================================

elif page == "📊 Analyse des risques":
    st.markdown(
        """
        <div class="section-title">
            📊 Analyse globale des risques
        </div>
        """,
        unsafe_allow_html=True
    )

    if "resultats_predictions" not in st.session_state:
        st.info(
            """
            Importez un fichier et lancez les prédictions
            pour accéder à cette analyse.
            """
        )

    else:
        resultats = st.session_state[
            "resultats_predictions"
        ]

        repartition = (
            resultats["Classe_Risque_PREDITE"]
            .value_counts()
            .reindex(
                ORDRE_RISQUE,
                fill_value=0
            )
        )

        total = len(resultats)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total GAB", total)

        with col2:
            st.metric(
                "Risque faible",
                int(repartition["Faible"])
            )

        with col3:
            st.metric(
                "Risque moyen",
                int(repartition["Moyen"])
            )

        with col4:
            st.metric(
                "Risque élevé",
                int(repartition["Élevé"])
            )

        st.markdown(
            "### Répartition des niveaux de risque"
        )

        st.bar_chart(repartition)

        if total > 0:
            pourcentages = repartition / total * 100

            tableau_repartition = pd.DataFrame({
                "Niveau de risque":
                    repartition.index,

                "Nombre de GAB":
                    repartition.values,

                "Pourcentage": [
                    f"{valeur:.2f} %"
                    for valeur in pourcentages.values
                ]
            })

            st.markdown("### Synthèse")

            st.dataframe(
                tableau_repartition,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            "### Probabilités moyennes prédites"
        )

        colonnes_probabilites = [
            colonne
            for colonne in resultats.columns
            if colonne.startswith("Prob_Risque_")
        ]

        if colonnes_probabilites:
            moyennes = (
                resultats[colonnes_probabilites]
                .mean()
                * 100
            )

            tableau_probabilites = pd.DataFrame({
                "Niveau de risque": [
                    colonne.replace(
                        "Prob_Risque_",
                        ""
                    )
                    for colonne in moyennes.index
                ],

                "Probabilité moyenne": [
                    f"{valeur:.2f} %"
                    for valeur in moyennes.values
                ]
            })

            st.dataframe(
                tableau_probabilites,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# AUTEUR
# ============================================================

elif page == "👩‍💻 Auteur":

    st.markdown("## 👩‍💻 À propos de l'auteur")
    st.markdown("---")

    col1, col2 = st.columns(
        [1, 2],
        gap="large"
    )

    # ========================================================
    # CARTE DE PRESENTATION
    # ========================================================

    with col1:

        with st.container(border=True):

            st.markdown(
                "<div style='text-align:center; font-size:75px;'>"
                "👩‍💻"
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<h2 style='text-align:center; margin-bottom:8px;'>"
                "Mariam KONATE"
                "</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='"
                "text-align:center;"
                "font-size:16px;"
                "font-weight:600;"
                "color:#374151;"
                "margin-bottom:8px;"
                "'>"
                "📊 Data Scientist"
                "</p>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='"
                "text-align:center;"
                "font-size:16px;"
                "font-weight:600;"
                "color:#374151;"
                "margin-bottom:8px;"
                "'>"
                "📈 Data Analyst"
                "</p>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='"
                "text-align:center;"
                "font-size:16px;"
                "font-weight:600;"
                "color:#374151;"
                "margin-bottom:0;"
                "'>"
                "🤖 Machine Learning"
                "</p>",
                unsafe_allow_html=True
            )

        st.markdown("### 📬 Contact")

        st.link_button(
            label="✉️ Envoyer un e-mail",
            url="mailto:konatemariam1203@gmail.com",
            use_container_width=True
        )

        st.link_button(
            label="🔗 Consulter le profil LinkedIn",
            url=(
                "https://www.linkedin.com/in/"
                "mariam-konat%C3%A9-881409348"
            ),
            use_container_width=True
        )

    # ========================================================
    # PARCOURS ET COMPETENCES
    # ========================================================

    with col2:

        st.markdown("### 🎓 Parcours académique")

        with st.container(border=True):

            st.markdown(
                """
                #### 🎓 Master en Data Science

                **Niveau actuel :** Master 2  
                **Spécialité :** Data Science/Machine Learning/Data Analytics
                """
            )

            st.markdown("---")

            st.markdown(
                """
                #### 📐 Licence en Mathématiques appliquées 
                **Diplôme :** Licence  
                **Spécialité :** Mathématiques & applications
                """
            )

        st.markdown("### 💼 Domaines d'intervention")

        domaine_col1, domaine_col2 = st.columns(2)

        with domaine_col1:

            st.markdown(
                """
                - 📊 **Data Science**
                - 📈 **Analyse de données**
                - 🤖 **Machine Learning**
                - 🔮 **Modélisation prédictive**
                """
            )

        with domaine_col2:

            st.markdown(
                """
                - 📐 **Analyse statistique**
                - 📉 **Visualisation de données**
                - ⚙️ **Automatisation des traitements**
                - 💡 **Aide à la décision**
                """
            )

        st.markdown("### 🛠️ Outils et technologies")

        outils_col1, outils_col2, outils_col3 = st.columns(3)

        with outils_col1:

            with st.container(border=True):

                st.markdown("#### 💻 Programmation")

                st.markdown(
                    """
                    - Python
                    - SQL
                    - R
                    """
                )

        with outils_col2:

            with st.container(border=True):

                st.markdown("#### 🤖 Analyse et modélisation")

                st.markdown(
                    """
                    - Pandas
                    - NumPy
                    - Scikit-learn
                    - LightGBM
                
                    """
                )

        with outils_col3:   

            with st.container(border=True):

                st.markdown("#### 📊 Visualisation")

                st.markdown(
                    """
                    - Matplotlib
                    - Seaborn
                    - Plotly
                    - Streamlit
                    """
                )

        
        st.info(
            """
            Profil orienté vers l'analyse et la valorisation
            des données, la modélisation prédictive et le
            développement de solutions d'aide à la décision.
            """
        )


# ============================================================
# PIED DE PAGE
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#777777;
        font-size:13px;
        padding:10px;
    ">
        Prédiction du risque de panne des GAB |
        OMOA Côte d'Ivoire |
        Master 2 Data Science
    </div>
    """,
    unsafe_allow_html=True
)