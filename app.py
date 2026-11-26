import os
import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Omutch — Estimateur Prix VO",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "vehicules.duckdb"

DEPARTEMENTS_FRANCE = {
    "01": "01 - Ain", "02": "02 - Aisne", "03": "03 - Allier", "04": "04 - Alpes-de-Haute-Provence",
    "05": "05 - Hautes-Alpes", "06": "06 - Alpes-Maritimes", "07": "07 - Ardèche", "08": "08 - Ardennes",
    "09": "09 - Ariège", "10": "10 - Aube", "11": "11 - Aude", "12": "12 - Aveyron",
    "13": "13 - Bouches-du-Rhône", "14": "14 - Calvados", "15": "15 - Cantal", "16": "16 - Charente",
    "17": "17 - Charente-Maritime", "18": "18 - Cher", "19": "19 - Corrèze", "2A": "2A - Corse-du-Sud",
    "2B": "2B - Haute-Corse", "21": "21 - Côte-d'Or", "22": "22 - Côtes-d'Armor", "23": "23 - Creuse",
    "24": "24 - Dordogne", "25": "25 - Doubs", "26": "26 - Drôme", "27": "27 - Eure",
    "28": "28 - Eure-et-Loir", "29": "29 - Finistère", "30": "30 - Gard", "31": "31 - Haute-Garonne",
    "32": "32 - Gers", "33": "33 - Gironde", "34": "34 - Hérault", "35": "35 - Ille-et-Vilaine",
    "36": "36 - Indre", "37": "37 - Indre-et-Loire", "38": "38 - Isère", "39": "39 - Jura",
    "40": "40 - Landes", "41": "41 - Loir-et-Cher", "42": "42 - Loire", "43": "43 - Haute-Loire",
    "44": "44 - Loire-Atlantique", "45": "45 - Loiret", "46": "46 - Lot", "47": "47 - Lot-et-Garonne",
    "48": "48 - Lozère", "49": "49 - Maine-et-Loire", "50": "50 - Manche", "51": "51 - Marne",
    "52": "52 - Haute-Marne", "53": "53 - Mayenne", "54": "54 - Meurthe-et-Moselle", "55": "55 - Meuse",
    "56": "56 - Morbihan", "57": "57 - Moselle", "58": "58 - Nièvre", "59": "59 - Nord",
    "60": "60 - Oise", "61": "61 - Orne", "62": "62 - Pas-de-Calais", "63": "63 - Puy-de-Dôme",
    "64": "64 - Pyrénées-Atlantiques", "65": "65 - Hautes-Pyrénées", "66": "66 - Pyrénées-Orientales",
    "67": "67 - Bas-Rhin", "68": "68 - Haut-Rhin", "69": "69 - Rhône", "70": "70 - Haute-Saône",
    "71": "71 - Saône-et-Loire", "72": "72 - Sarthe", "73": "73 - Savoie", "74": "74 - Haute-Savoie",
    "75": "75 - Paris", "76": "76 - Seine-Maritime", "77": "77 - Seine-et-Marne", "78": "78 - Yvelines",
    "79": "79 - Deux-Sèvres", "80": "80 - Somme", "81": "81 - Tarn", "82": "82 - Tarn-et-Garonne",
    "83": "83 - Var", "84": "84 - Vaucluse", "85": "85 - Vendée", "86": "86 - Vienne",
    "87": "87 - Haute-Vienne", "88": "88 - Vosges", "89": "89 - Yonne", "90": "90 - Territoire de Belfort",
    "91": "91 - Essonne", "92": "92 - Hauts-de-Seine", "93": "93 - Seine-Saint-Denis",
    "94": "94 - Val-de-Marne", "95": "95 - Val-d'Oise",
    "971": "971 - Guadeloupe", "972": "972 - Martinique", "973": "973 - Guyane",
    "974": "974 - La Réunion", "976": "976 - Mayotte",
}

ZONES_ATYPIQUES = {
    "Grandes Métropoles (+200k hab. & périphérie)": {
        "75", "92", "93", "94", "69", "13", "31", "06", "44", "34", "67", "33", "59", "35"
    },
    "Montagne (Alpes, Pyrénées, Jura, Massif Central)": {
        "04", "05", "09", "15", "25", "38", "39", "43", "48", "63", "64", "65", "66", "73", "74"
    },
    "Côte Atlantique & Manche": {
        "14", "17", "22", "29", "33", "40", "44", "50", "56", "64", "76", "85"
    },
    "Côtes Méditerranéennes & Corse": {
        "06", "11", "13", "2A", "2B", "30", "34", "66", "83"
    },
    "Zones Frontalières (Suisse, Luxembourg, Allemagne, Belgique, Espagne)": {
        "01", "08", "25", "39", "54", "57", "59", "62", "64", "65", "66", "67", "68", "74", "90"
    },
    "DOM-TOM (Outre-Mer)": {
        "971", "972", "973", "974", "976"
    },
}


def extraire_code_dept(cp: str) -> str:
    if not cp or cp == "00000":
        return ""
    cp = str(cp).strip().upper()
    if cp.startswith(("2A", "2B")):
        return cp[:2]
    if cp.startswith("97") and len(cp) >= 3:
        return cp[:3]
    if len(cp) >= 2 and cp[:2].isdigit():
        return cp[:2]
    return ""


def classifier_zone_atypique(cp: str) -> str:
    dept = extraire_code_dept(cp)
    if not dept:
        return "Non renseigné"
    if dept in ZONES_ATYPIQUES["DOM-TOM (Outre-Mer)"]:
        return "DOM-TOM (Outre-Mer)"
    if dept in ZONES_ATYPIQUES["Montagne (Alpes, Pyrénées, Jura, Massif Central)"]:
        return "Montagne"
    if dept in ZONES_ATYPIQUES["Grandes Métropoles (+200k hab. & périphérie)"]:
        return "Grandes Métropoles (+200k)"
    if dept in ZONES_ATYPIQUES["Côtes Méditerranéennes & Corse"]:
        return "Côtes Méditerranéennes"
    if dept in ZONES_ATYPIQUES["Côte Atlantique & Manche"]:
        return "Côte Atlantique"
    if dept in ZONES_ATYPIQUES["Zones Frontalières (Suisse, Luxembourg, Allemagne, Belgique, Espagne)"]:
        return "Zones Frontalières"
    return "Campagne / Rural"


def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=60)
def charger_filtres_initiaux():
    con = get_connection()
    marques = [
        r[0]
        for r in con.execute(
            """
            SELECT DISTINCT marque FROM vehicules
            WHERE marque IS NOT NULL AND marque != 'INCONNU'
            ORDER BY marque
            """
        ).fetchall()
    ]
    sources = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT source FROM annonces WHERE source IS NOT NULL ORDER BY source"
        ).fetchall()
    ]
    depts_raw = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT code_postal FROM vehicules WHERE code_postal IS NOT NULL AND code_postal != '00000'"
        ).fetchall()
    ]
    total_annonces = con.execute("SELECT COUNT(*) FROM annonces").fetchone()[0]
    con.close()

    depts_presents = sorted({extraire_code_dept(cp) for cp in depts_raw if extraire_code_dept(cp)})
    return marques, sources, depts_presents, total_annonces


@st.cache_data(ttl=60)
def charger_composition_dataset():
    """Charge les statistiques globales de composition de la base de données."""
    con = get_connection()
    df_sources = con.execute(
        """
        SELECT 
            UPPER(source) AS Plateforme,
            COUNT(*) AS Annonces,
            MIN(date_publication) AS Premiere_collecte,
            MAX(date_derniere_vue) AS Derniere_collecte
        FROM annonces
        GROUP BY UPPER(source)
        ORDER BY Annonces DESC
        """
    ).df()

    df_energies = con.execute(
        """
        SELECT 
            COALESCE(NULLIF(LOWER(v.energie), ''), 'inconnu') AS Energie,
            COUNT(*) AS Annonces
        FROM annonces a
        JOIN vehicules v ON a.id_vehicule = v.id_vehicule
        GROUP BY 1
        ORDER BY Annonces DESC
        """
    ).df()

    df_boites = con.execute(
        """
        SELECT 
            COALESCE(NULLIF(LOWER(v.boite_vitesse), ''), 'inconnu') AS Boite,
            COUNT(*) AS Annonces
        FROM annonces a
        JOIN vehicules v ON a.id_vehicule = v.id_vehicule
        GROUP BY 1
        ORDER BY Annonces DESC
        """
    ).df()

    df_annees = con.execute(
        """
        SELECT 
            v.annee_mise_circulation AS Annee,
            a.source AS Source,
            COUNT(*) AS Annonces
        FROM annonces a
        JOIN vehicules v ON a.id_vehicule = v.id_vehicule
        WHERE v.annee_mise_circulation BETWEEN 2005 AND 2026
        GROUP BY 1, 2
        ORDER BY 1
        """
    ).df()

    df_top_marques = con.execute(
        """
        SELECT v.marque AS Marque, COUNT(*) AS Annonces
        FROM annonces a
        JOIN vehicules v ON a.id_vehicule = v.id_vehicule
        WHERE v.marque IS NOT NULL AND v.marque != 'INCONNU'
        GROUP BY v.marque
        ORDER BY Annonces DESC
        LIMIT 30
        """
    ).df()

    nb_vehicules = con.execute("SELECT COUNT(*) FROM vehicules").fetchone()[0]
    nb_releves = con.execute("SELECT COUNT(*) FROM historique_prix").fetchone()[0]
    con.close()

    return df_sources, df_energies, df_boites, df_annees, df_top_marques, nb_vehicules, nb_releves


@st.cache_data(ttl=60)
def charger_options_marque(marque):
    con = get_connection()
    if marque == "Toutes":
        modeles = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT modele FROM vehicules WHERE modele IS NOT NULL AND modele != 'INCONNU' ORDER BY modele"
            ).fetchall()
        ]
        energies = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT energie FROM vehicules WHERE energie IS NOT NULL ORDER BY energie"
            ).fetchall()
        ]
        boites = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT boite_vitesse FROM vehicules WHERE boite_vitesse IS NOT NULL ORDER BY boite_vitesse"
            ).fetchall()
        ]
    else:
        modeles = [
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT modele FROM vehicules
                WHERE UPPER(marque) = UPPER(?) AND modele IS NOT NULL AND modele != 'INCONNU'
                ORDER BY modele
                """,
                [marque],
            ).fetchall()
        ]
        energies = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT energie FROM vehicules WHERE UPPER(marque) = UPPER(?) AND energie IS NOT NULL ORDER BY energie",
                [marque],
            ).fetchall()
        ]
        boites = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT boite_vitesse FROM vehicules WHERE UPPER(marque) = UPPER(?) AND boite_vitesse IS NOT NULL ORDER BY boite_vitesse",
                [marque],
            ).fetchall()
        ]
    con.close()
    return modeles, energies, boites


@st.cache_data(ttl=30)
def rechercher_comparables(
    marque, modele, energie, boite, source, annee_min, annee_max, inclure_vendus
):
    con = get_connection()
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'vehicules'"
        ).fetchall()
    }
    col_ville = "v.ville," if "ville" in cols else "'' AS ville,"

    query = f"""
        WITH derniers_releves AS (
            SELECT 
                id_annonce,
                prix_affiche,
                kilometrage,
                date_releve,
                ROW_NUMBER() OVER (PARTITION BY id_annonce ORDER BY date_releve DESC) AS rn
            FROM historique_prix
        )
        SELECT 
            a.source,
            a.statut,
            v.marque, 
            v.modele, 
            v.energie,
            v.boite_vitesse AS boite,
            v.annee_mise_circulation AS annee, 
            hp.kilometrage, 
            hp.prix_affiche AS prix, 
            v.code_postal,
            {col_ville}
            hp.date_releve,
            a.url_annonce
        FROM annonces a
        JOIN vehicules v ON a.id_vehicule = v.id_vehicule
        JOIN derniers_releves hp ON a.id_annonce = hp.id_annonce AND hp.rn = 1
        WHERE hp.prix_affiche BETWEEN 500 AND 300000
          AND hp.kilometrage BETWEEN 0 AND 450000
    """
    params = []

    if not inclure_vendus:
        query += " AND COALESCE(a.statut, 'active') = 'active'"

    if marque != "Toutes":
        query += " AND UPPER(v.marque) = UPPER(?)"
        params.append(marque)

    if modele != "Tous":
        query += " AND UPPER(v.modele) LIKE UPPER(?)"
        params.append(f"%{modele}%")

    if energie != "Tous":
        query += " AND LOWER(v.energie) = LOWER(?)"
        params.append(energie.lower())

    if boite != "Toutes":
        query += " AND LOWER(v.boite_vitesse) = LOWER(?)"
        params.append(boite.lower())

    if source != "Toutes":
        query += " AND LOWER(a.source) = LOWER(?)"
        params.append(source.lower())

    if annee_min is not None and annee_max is not None:
        query += " AND v.annee_mise_circulation BETWEEN ? AND ?"
        params.extend([annee_min, annee_max])

    df = con.execute(query, params).df()
    con.close()

    if not df.empty:
        df["dept"] = df["code_postal"].apply(extraire_code_dept)
        df["zone_atypique"] = df["code_postal"].apply(classifier_zone_atypique)
    return df


# --- EN-TÊTE ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=110)
with col_title:
    st.title("Omutch — Estimateur de prix véhicules d'occasion")
    st.caption("Analyse statistique et comparables en temps réel du marché français")

marques_dispo, sources_dispo, depts_dispo, total_base = charger_filtres_initiaux()

# --- BARRE LATÉRALE ---
st.sidebar.header("🔍 Critères du véhicule")
st.sidebar.metric("Annonces en base", f"{total_base:,}".replace(",", " "))

selected_marque = st.sidebar.selectbox("Marque", ["Toutes"] + marques_dispo)
modeles_dispo, energies_dispo, boites_dispo = charger_options_marque(selected_marque)

selected_modele = st.sidebar.selectbox("Modèle", ["Tous"] + modeles_dispo)
selected_energie = st.sidebar.selectbox("Énergie", ["Tous"] + energies_dispo)
selected_boite = st.sidebar.selectbox("Boîte de vitesse", ["Toutes"] + boites_dispo)

# Année optionnelle
filtrer_annee = st.sidebar.checkbox("Filtrer par année", value=False)
if filtrer_annee:
    col_a1, col_a2 = st.sidebar.columns(2)
    with col_a1:
        annee_cible = st.number_input("Année", min_value=1990, max_value=2026, value=2020, step=1)
    with col_a2:
        tolerance_annee = st.selectbox("Tolérance", [0, 1, 2, 3, 5], index=1, format_func=lambda x: f"± {x} an(s)")
    annee_min_query = annee_cible - tolerance_annee
    annee_max_query = annee_cible + tolerance_annee
else:
    annee_min_query = None
    annee_max_query = None

# Kilométrage optionnel
ajuster_km = st.sidebar.checkbox("Ajuster à un kilométrage précis", value=False)
if ajuster_km:
    kilometrage_cible = st.sidebar.number_input(
        "Kilométrage cible (km)", min_value=0, max_value=400000, value=85000, step=5000
    )
else:
    kilometrage_cible = None

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Segmentation géographique")

mode_geo = st.sidebar.radio(
    "Découpage territorial",
    [
        "France entière",
        "Par Département",
        "Par Zonage Atypique",
        "Par Ville / Code Postal",
    ],
)

dept_choisi = None
zone_atypique_choisie = None
recherche_libre_geo = ""

if mode_geo == "Par Département":
    options_depts = [
        DEPARTEMENTS_FRANCE.get(d, f"{d} - Département {d}") for d in depts_dispo
    ]
    dept_label = st.sidebar.selectbox("Sélectionner un département", options_depts)
    if dept_label:
        dept_choisi = dept_label.split(" - ")[0].strip()

elif mode_geo == "Par Zonage Atypique":
    liste_zones_filtre = [
        "Grandes Métropoles (+200k)",
        "Montagne",
        "Côte Atlantique",
        "Côtes Méditerranéennes",
        "Zones Frontalières",
        "Campagne / Rural",
        "DOM-TOM (Outre-Mer)",
    ]
    zone_atypique_choisie = st.sidebar.selectbox("Typologie de zone", liste_zones_filtre)

elif mode_geo == "Par Ville / Code Postal":
    recherche_libre_geo = st.sidebar.text_input(
        "Ville ou Code Postal", placeholder="Ex: Rennes, 35, 74000, Annecy..."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Paramètres avancés")
selected_source = st.sidebar.selectbox("Plateforme source", ["Toutes"] + sources_dispo)
filtrer_outliers = st.sidebar.checkbox("Exclure les prix extrêmes (IQR)", value=True)
inclure_vendus = st.sidebar.checkbox("Inclure les annonces retirées/vendues", value=False)

# --- CHARGEMENT & FILTRAGE DES DONNÉES ---
df_brut = rechercher_comparables(
    selected_marque,
    selected_modele,
    selected_energie,
    selected_boite,
    selected_source,
    annee_min_query,
    annee_max_query,
    inclure_vendus,
)

df_france = df_brut.copy()

df = df_brut.copy()
if not df.empty:
    if mode_geo == "Par Département" and dept_choisi:
        df = df[df["dept"] == dept_choisi]
    elif mode_geo == "Par Zonage Atypique" and zone_atypique_choisie:
        df = df[df["zone_atypique"] == zone_atypique_choisie]
    elif mode_geo == "Par Ville / Code Postal" and recherche_libre_geo.strip():
        terme = recherche_libre_geo.strip()
        if terme.isdigit():
            df = df[df["code_postal"].astype(str).str.startswith(terme)]
        else:
            df = df[df["ville"].fillna("").str.contains(terme, case=False, na=False)]

if filtrer_outliers and len(df) >= 8:
    q1 = df["prix"].quantile(0.25)
    q3 = df["prix"].quantile(0.75)
    iqr = q3 - q1
    df = df[(df["prix"] >= q1 - 1.5 * iqr) & (df["prix"] <= q3 + 1.5 * iqr)]

# --- ONGLETS PRINCIPAUX ---
tab_estim, tab_marche, tab_liste = st.tabs(
    ["💰 Estimation & Cote", "📊 Analyse du Marché & Dataset", "📋 Annonces Comparables"]
)

with tab_estim:
    if df.empty:
        st.warning("Aucune annonce comparable trouvée avec ces critères. Élargissez vos filtres ou la zone géographique.")
    else:
        prix_median = df["prix"].median()
        prix_bas = df["prix"].quantile(0.25)
        prix_haut = df["prix"].quantile(0.75)
        km_moyen = df["kilometrage"].mean()
        nb_fmt = f"{len(df):,}".replace(",", " ")

        if ajuster_km and kilometrage_cible is not None:
            decote_par_km = 0.08
            methode_decote = "standard : -80 € / 1 000 km"
            if len(df) >= 10 and df["kilometrage"].std() > 5000:
                pente, _ = np.polyfit(df["kilometrage"], df["prix"], 1)
                if -0.25 <= pente <= -0.01:
                    decote_par_km = abs(pente)
                    methode_decote = f"tendance marché : -{decote_par_km * 1000:,.0f} € / 1 000 km".replace(",", " ")

            prix_conseille = max(500, prix_median - (kilometrage_cible - km_moyen) * decote_par_km)
            ratio_ajuste = prix_conseille / prix_median if prix_median > 0 else 1.0
            prix_bas_affiche = max(500, prix_bas * ratio_ajuste)
            prix_haut_affiche = max(500, prix_haut * ratio_ajuste)

            km_fmt = f"{kilometrage_cible:,}".replace(",", " ")
            st.info(
                f"**Prix conseillé ajusté à {km_fmt} km** ({methode_decote}) • Basé sur **{nb_fmt} annonces uniques**"
            )
            label_median = "🎯 Prix médian ajusté"
        else:
            prix_conseille = prix_median
            prix_bas_affiche = prix_bas
            prix_haut_affiche = prix_haut
            st.info(
                f"**Cote globale du marché** (toutes années/kilométrages sélectionnés) • Basé sur **{nb_fmt} annonces uniques**"
            )
            label_median = "🎯 Prix médian du marché"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fourchette basse (Q1 - 25%)", f"{prix_bas_affiche:,.0f} €".replace(",", " "))
        c2.metric(label_median, f"{prix_conseille:,.0f} €".replace(",", " "))
        c3.metric("Fourchette haute (Q3 - 75%)", f"{prix_haut_affiche:,.0f} €".replace(",", " "))
        c4.metric("Kilométrage moyen constaté", f"{km_moyen:,.0f} km".replace(",", " "))

        st.subheader("📉 Décote kilométrique des véhicules comparables")
        echantillon_graph = df.sample(min(len(df), 1500), random_state=42)
        scatter = (
            alt.Chart(echantillon_graph)
            .mark_circle(size=55, opacity=0.55)
            .encode(
                x=alt.X("kilometrage:Q", title="Kilométrage (km)"),
                y=alt.Y("prix:Q", title="Prix affiché (€)"),
                color=alt.Color("source:N", title="Source"),
                tooltip=["marque", "modele", "annee", "kilometrage", "prix", "ville", "code_postal", "source"],
            )
        )
        if ajuster_km and kilometrage_cible is not None:
            point_cible = (
                alt.Chart(pd.DataFrame({"kilometrage": [kilometrage_cible], "prix": [prix_conseille]}))
                .mark_point(color="red", size=220, filled=True, shape="diamond")
                .encode(
                    x="kilometrage:Q",
                    y="prix:Q",
                    tooltip=[alt.Tooltip("prix:Q", title="Estimation (€)", format=",.0f")],
                )
            )
            st.altair_chart(scatter + point_cible, use_container_width=True)
        else:
            st.altair_chart(scatter, use_container_width=True)

with tab_marche:
    # 1. Comparatif par Zonage Atypique sur la sélection active
    st.subheader("🗺️ Comparatif des prix médians par Zonage Atypique (France entière)")
    df_zones = df_france[df_france["zone_atypique"] != "Non renseigné"].copy()
    if len(df_zones) >= 5:
        stats_zones = (
            df_zones.groupby("zone_atypique", as_index=False)
            .agg(
                Prix_Median=("prix", "median"),
                Km_Moyen=("kilometrage", "mean"),
                Annonces=("prix", "count"),
            )
            .sort_values("Prix_Median", ascending=False)
        )
        chart_zones = (
            alt.Chart(stats_zones)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("zone_atypique:N", sort="-y", title="Zone typologique"),
                y=alt.Y("Prix_Median:Q", title="Prix médian (€)"),
                color=alt.Color("zone_atypique:N", legend=None),
                tooltip=[
                    alt.Tooltip("zone_atypique:N", title="Zone"),
                    alt.Tooltip("Prix_Median:Q", title="Prix médian (€)", format=",.0f"),
                    alt.Tooltip("Km_Moyen:Q", title="Km moyen", format=",.0f"),
                    alt.Tooltip("Annonces:Q", title="Nb annonces"),
                ],
            )
        )
        st.altair_chart(chart_zones, use_container_width=True)
        st.dataframe(
            stats_zones.rename(
                columns={
                    "zone_atypique": "Zone atypique",
                    "Prix_Median": "Prix médian (€)",
                    "Km_Moyen": "Kilométrage moyen",
                    "Annonces": "Nombre d'annonces",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Pas assez d'annonces géolocalisées sur cette sélection pour comparer les zones atypiques.")

    st.markdown("---")

    # 2. Composition globale du dataset
    st.subheader("🗄️ Composition globale du dataset Omutch")
    (
        df_sources,
        df_energies,
        df_boites,
        df_annees,
        df_top_marques,
        nb_vehicules,
        nb_releves,
    ) = charger_composition_dataset()

    k1, k2, k3 = st.columns(3)
    k1.metric("Annonces uniques suivies", f"{total_base:,}".replace(",", " "))
    k2.metric("Véhicules physiques dédupliqués", f"{nb_vehicules:,}".replace(",", " "))
    k3.metric("Relevés de prix historisés", f"{nb_releves:,}".replace(",", " "))

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("**Répartition par plateforme**")
        st.dataframe(
            df_sources.rename(
                columns={
                    "Premiere_collecte": "Début collecte",
                    "Derniere_collecte": "Dernière maj",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    with col_d2:
        st.markdown("**Répartition par motorisation**")
        chart_en = (
            alt.Chart(df_energies)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Energie:N", sort="-y", title="Énergie"),
                y=alt.Y("Annonces:Q", title="Annonces"),
                color=alt.Color("Energie:N", legend=None),
                tooltip=["Energie", "Annonces"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart_en, use_container_width=True)
    with col_d3:
        st.markdown("**Répartition par boîte de vitesse**")
        chart_boite = (
            alt.Chart(df_boites)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Boite:N", sort="-y", title="Boîte"),
                y=alt.Y("Annonces:Q", title="Annonces"),
                color=alt.Color("Boite:N", legend=None),
                tooltip=["Boite", "Annonces"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart_boite, use_container_width=True)

    st.markdown("**Distribution des annonces par année de mise en circulation (2005 – 2026)**")
    chart_annees = (
        alt.Chart(df_annees)
        .mark_bar()
        .encode(
            x=alt.X("Annee:O", title="Année"),
            y=alt.Y("Annonces:Q", title="Nombre d'annonces"),
            color=alt.Color("Source:N", title="Plateforme"),
            tooltip=["Annee", "Source", "Annonces"],
        )
    )
    st.altair_chart(chart_annees, use_container_width=True)

    st.subheader("🏆 Top 30 des marques en base (par volume)")
    chart_top = (
        alt.Chart(df_top_marques)
        .mark_bar()
        .encode(
            x=alt.X("Marque:N", sort="-y", title="Marque"),
            y=alt.Y("Annonces:Q", title="Nombre d'annonces uniques"),
            tooltip=["Marque", "Annonces"],
        )
    )
    st.altair_chart(chart_top, use_container_width=True)

with tab_liste:
    nb_res = f"{len(df):,}".replace(",", " ")
    st.subheader(f"Annonces comparables ({nb_res} résultats, triées par prix croissant)")
    if not df.empty:
        df_affichage = df.sort_values("prix", ascending=True).head(500)[
            ["source", "marque", "modele", "annee", "kilometrage", "prix", "energie", "boite", "ville", "code_postal", "zone_atypique", "url_annonce"]
        ]
        st.dataframe(
            df_affichage,
            hide_index=True,
            use_container_width=True,
            column_config={
                "source": "Source",
                "marque": "Marque",
                "modele": "Modèle",
                "annee": st.column_config.NumberColumn("Année", format="%d"),
                "kilometrage": st.column_config.NumberColumn("Kilométrage", format="%d km"),
                "prix": st.column_config.NumberColumn("Prix", format="%d €"),
                "energie": "Énergie",
                "boite": "Boîte",
                "ville": "Ville",
                "code_postal": "CP",
                "zone_atypique": "Zone atypique",
                "url_annonce": st.column_config.LinkColumn("Lien annonce", display_text="Voir l'annonce"),
            },
        )