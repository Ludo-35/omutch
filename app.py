import os
import urllib.parse
import altair as alt
import duckdb
import numpy as np
import pandas as pd
import requests
import streamlit as st

LOGO_PATH = "logo.png"
DB_PATH = "vehicules.duckdb"

# Configuration de la page
st.set_page_config(
    page_title="Omutch • Estimateur Auto",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

# Style CSS léger pour moderniser les cartes et l'affichage
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 14px 18px;
        border-radius: 10px;
    }
    .highlight-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .highlight-card h3 { margin: 0; font-size: 0.95rem; opacity: 0.85; font-weight: 500; }
    .highlight-card h1 { margin: 4px 0 0 0; font-size: 2.3rem; font-weight: 700; }
    .highlight-card p { margin: 6px 0 0 0; font-size: 0.85rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data(ttl=60)
def charger_referentiels():
    """Charge les listes de filtres et compteurs globaux (mis en cache 60s)."""
    con = get_connection()
    total = con.execute("SELECT COUNT(*) FROM annonces").fetchone()[0]

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

    energies = [
        r[0].capitalize()
        for r in con.execute(
            """
            SELECT DISTINCT energie FROM vehicules 
            WHERE energie IS NOT NULL AND energie != 'inconnu' 
            ORDER BY energie
            """
        ).fetchall()
    ]

    boites = [
        r[0].capitalize()
        for r in con.execute(
            """
            SELECT DISTINCT boite_vitesse FROM vehicules 
            WHERE boite_vitesse IS NOT NULL AND boite_vitesse != 'inconnu' 
            ORDER BY boite_vitesse
            """
        ).fetchall()
    ]

    con.close()
    return total, marques, energies, boites


@st.cache_data(ttl=60)
def charger_modeles(marque: str):
    con = get_connection()
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
    con.close()
    return modeles


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
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def recuperer_photos_generations(
    marque: str, modele: str, annees_cibles: list[int]
) -> list[dict]:
    """Récupère les photos Wikimedia Commons correspondant aux années/générations du véhicule."""
    if not marque or marque == "Toutes":
        return []

    modele_clean = "" if modele == "Tous" else modele
    annees_uniques = sorted(set(a for a in annees_cibles if a and 1980 <= a <= 2026))
    if not annees_uniques:
        annees_uniques = [None]

    headers = {"User-Agent": "OmutchAutoApp/1.0 (contact@omutch.local)"}
    photos = []
    urls_vues = set()

    for an in annees_uniques:
        terme = f"{marque} {modele_clean} {an if an else ''}".strip()
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {terme}",
                    "gsrnamespace": 6,
                    "gsrlimit": 4,
                    "prop": "imageinfo",
                    "iiprop": "url",
                    "iiurlwidth": 480,
                    "format": "json",
                },
                headers=headers,
                timeout=4,
            )
            pages = resp.json().get("query", {}).get("pages", {})
            for page_data in pages.values():
                titre_fichier = page_data.get("title", "").lower()
                if any(
                    mot in titre_fichier
                    for mot in [
                        "logo",
                        "badge",
                        "emblem",
                        "interior",
                        "dashboard",
                        "engine",
                        "diagram",
                    ]
                ):
                    continue
                ii = page_data.get("imageinfo", [{}])[0]
                thumb_url = ii.get("thumburl") or ii.get("url")
                if thumb_url and thumb_url not in urls_vues:
                    urls_vues.add(thumb_url)
                    legende = f"{marque} {modele_clean}".strip() + (
                        f" ({an})" if an else ""
                    )
                    photos.append({"url": thumb_url, "legende": legende})
                    break
        except Exception:
            continue

    return photos


# --- EN-TÊTE ---
col_logo, col_title, col_refresh = st.columns(
    [1.2, 4.8, 1], vertical_alignment="center"
)
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)
with col_title:
    st.title("Cockpit d'Estimation Véhicule")
with col_refresh:
    if st.button(
        "🔄 Actualiser",
        use_container_width=True,
        help="Recharger les dernières annonces scrappées",
    ):
        st.cache_data.clear()
        st.rerun()

# Chargement des référentiels
try:
    total_annonces, marques_dispo, energies_dispo, boites_dispo = charger_referentiels()
except Exception as e:
    st.error(f"Impossible de lire la base de données (`{DB_PATH}`) : {e}")
    st.stop()

# --- BARRE LATÉRALE (FILTRES) ---
st.sidebar.metric("📊 Annonces en base", f"{total_annonces:,}")
st.sidebar.markdown("---")
st.sidebar.subheader("1. Véhicule à évaluer")

selected_marque = st.sidebar.selectbox("Marque", options=["Toutes"] + marques_dispo)

if selected_marque != "Toutes":
    modeles_dispo = ["Tous"] + charger_modeles(selected_marque)
else:
    modeles_dispo = ["Tous"]
selected_modele = st.sidebar.selectbox("Modèle", options=modeles_dispo)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    selected_energie = st.selectbox("Énergie", options=["Tous"] + energies_dispo)
with col_sb2:
    selected_boite = st.selectbox("Boîte", options=["Toutes"] + boites_dispo)

selected_source = st.sidebar.selectbox(
    "Source des annonces", options=["Toutes", "Lacentrale", "Leboncoin"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Ajustements (Optionnels)")

filtrer_annee = st.sidebar.toggle("Filtrer par année", value=False)
if filtrer_annee:
    annee = st.sidebar.number_input("Année", 1995, 2026, 2019)
    marge_annee = st.sidebar.slider("Tolérance (± ans)", 0, 5, 1)
    annee_min, annee_max = annee - marge_annee, annee + marge_annee
else:
    annee, annee_min, annee_max = None, None, None

ajuster_km = st.sidebar.toggle("Ajuster selon le kilométrage", value=False)
if ajuster_km:
    kilometrage = st.sidebar.number_input(
        "Kilométrage actuel (km)", 0, 450000, 110000, step=5000
    )
else:
    kilometrage = None

exclure_outliers = st.sidebar.toggle("Écarter les prix aberrants", value=True)
inclure_vendus = st.sidebar.toggle("Inclure les véhicules vendus", value=False)

# --- EXÉCUTION DE LA RECHERCHE ---
df = rechercher_comparables(
    selected_marque,
    selected_modele,
    selected_energie,
    selected_boite,
    selected_source,
    annee_min,
    annee_max,
    inclure_vendus,
)

# --- ONGLETS PRINCIPAUX ---
tab_est, tab_analyse, tab_db = st.tabs(
    ["🎯 Estimateur & Comparables", "📈 Analyse du Marché", "🗄️ État de la Base"]
)

# ==============================================================================
# ONGLET 1 : ESTIMATEUR
# ==============================================================================
with tab_est:
    if df.empty:
        st.info(
            "Aucune annonce ne correspond exactement à ces critères. Élargis la tolérance d'année ou repasse un filtre sur « Tous »."
        )
    else:
        nb_brut = len(df)
        if exclure_outliers and nb_brut >= 10:
            q1, q3 = df["prix"].quantile(0.25), df["prix"].quantile(0.75)
            iqr = q3 - q1
            df = df[
                (df["prix"] >= max(500, q1 - 1.5 * iqr))
                & (df["prix"] <= q3 + 1.5 * iqr)
            ].copy()
        else:
            df = df.copy()

        nb_final = len(df)
        ecartes = nb_brut - nb_final

        prix_min = df["prix"].min()
        prix_max = df["prix"].max()
        prix_moyen = df["prix"].mean()
        prix_median = df["prix"].median()
        prix_p25 = df["prix"].quantile(0.25)
        prix_p75 = df["prix"].quantile(0.75)

        decote_par_km = 0.04
        methode_decote = "forfait standard 0,04 €/km"
        if ajuster_km and kilometrage is not None:
            if len(df) >= 10 and df["kilometrage"].std() > 5000:
                pente, _ = np.polyfit(df["kilometrage"], df["prix"], 1)
                if -0.25 <= pente <= -0.01:
                    decote_par_km = abs(pente)
                    methode_decote = f"tendance marché {decote_par_km:.3f} €/km"
            km_moyen = df["kilometrage"].mean()
            prix_conseille = max(
                500, prix_median - (kilometrage - km_moyen) * decote_par_km
            )
            sous_titre_carte = f"Médiane ajustée à {kilometrage:,} km ({methode_decote}) • Basé sur {nb_final:,} annonces"
        else:
            prix_conseille = prix_median
            sous_titre_carte = f"Prix médian du marché • Basé sur {nb_final:,} annonces" + (
                f" ({ecartes} extrêmes écartés)" if ecartes > 0 else ""
            )

        titre_vehicule = f"{selected_marque if selected_marque != 'Toutes' else 'Toutes marques'} {selected_modele if selected_modele != 'Tous' else ''}".strip()
        st.markdown(
            f"""
            <div class="highlight-card">
                <h3>ESTIMATION CONSEILLÉE • {titre_vehicule.upper()}</h3>
                <h1>{prix_conseille:,.0f} €</h1>
                <p>{sous_titre_carte}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Galerie des générations & liens presse/vidéo dès qu'une marque est choisie
        if selected_marque != "Toutes":
            annees_valides = df[df["annee"] > 1980]["annee"]
            if filtrer_annee and annee_min is not None and annee_max is not None:
                cibles_annees = (
                    [annee_min, annee_max] if annee_min != annee_max else [annee_min]
                )
            elif not annees_valides.empty:
                an_min_df, an_max_df = int(annees_valides.min()), int(
                    annees_valides.max()
                )
                cibles_annees = (
                    [an_min_df, an_max_df]
                    if (an_max_df - an_min_df) >= 4
                    else [int(annees_valides.median())]
                )
            else:
                cibles_annees = []

            photos_gen = recuperer_photos_generations(
                selected_marque, selected_modele, cibles_annees
            )
            if photos_gen:
                cols_img = st.columns(min(len(photos_gen), 3))
                for idx_p, photo in enumerate(photos_gen[:3]):
                    with cols_img[idx_p]:
                        st.image(
                            photo["url"],
                            caption=photo["legende"],
                            use_container_width=True,
                        )

            requete_recherche = f"{selected_marque} {selected_modele if selected_modele != 'Tous' else ''} {annee if annee else ''}".strip()
            q_enc = urllib.parse.quote_plus(requete_recherche)
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.link_button(
                    "🎬 Essais vidéo YouTube",
                    f"https://www.youtube.com/results?search_query=essai+{q_enc}",
                    use_container_width=True,
                )
            with col_l2:
                st.link_button(
                    "📰 Essai presse Caradisiac",
                    f"https://www.google.com/search?q=site:caradisiac.com+essai+{q_enc}",
                    use_container_width=True,
                )
            with col_l3:
                st.link_button(
                    "🔧 Fiche fiabilité & Avis",
                    f"https://www.google.com/search?q=fiabilite+avis+{q_enc}",
                    use_container_width=True,
                )

        # Indicateurs secondaires
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🔴 Vente rapide (P25)",
            f"{prix_p25:,.0f} €",
            help="25% des annonces sont sous ce prix",
        )
        c2.metric("📊 Prix moyen", f"{prix_moyen:,.0f} €")
        c3.metric(
            "🟢 Prix fort (P75)",
            f"{prix_p75:,.0f} €",
            help="75% des annonces sont sous ce prix",
        )
        c4.metric(
            "↔️ Extrêmes (Min / Max)", f"{prix_min:,.0f} € – {prix_max:,.0f} €"
        )

        st.markdown("---")

        if ajuster_km and kilometrage is not None:
            df["ecart_km"] = (df["kilometrage"] - kilometrage).abs()
            if filtrer_annee and annee is not None:
                df["score_proximite"] = (
                    df["ecart_km"] + (df["annee"] - annee).abs() * 15000
                )
            else:
                df["score_proximite"] = df["ecart_km"]
            df_affiche = df.sort_values("score_proximite").drop(
                columns=["ecart_km", "score_proximite"]
            )
            label_tri = "triées par proximité avec ton véhicule"
        else:
            df_affiche = df.sort_values("prix")
            label_tri = "triées par prix croissant"

        col_tbl_title, col_export = st.columns([4, 1])
        with col_tbl_title:
            st.subheader(
                f"📋 Annonces comparables ({nb_final:,} résultats, {label_tri})"
            )
        with col_export:
            csv_data = df_affiche.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                "📥 Exporter CSV",
                data=csv_data,
                file_name=f"comparables_{selected_marque.lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            df_affiche,
            column_config={
                "url_annonce": st.column_config.LinkColumn(
                    "Annonce", display_text="Ouvrir ↗"
                ),
                "source": st.column_config.TextColumn("Source"),
                "statut": st.column_config.TextColumn("Statut"),
                "marque": "Marque",
                "modele": "Modèle",
                "energie": "Énergie",
                "boite": "Boîte",
                "annee": st.column_config.NumberColumn("Année", format="%d"),
                "kilometrage": st.column_config.NumberColumn(
                    "Kilométrage", format="%d km"
                ),
                "prix": st.column_config.NumberColumn("Prix", format="%.0f €"),
                "code_postal": "CP",
                "ville": "Ville",
                "date_releve": st.column_config.DatetimeColumn(
                    "Relevé le", format="DD/MM/YYYY"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# ONGLET 2 : ANALYSE DU MARCHÉ
# ==============================================================================
with tab_analyse:
    if df.empty:
        st.info(
            "Sélectionne des critères avec des résultats pour afficher les graphiques d'analyse."
        )
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📉 Décote Prix vs Kilométrage")
            st.scatter_chart(df, x="kilometrage", y="prix", color="source")

        with col_g2:
            st.subheader("🏷️ Comparaison La Centrale vs Leboncoin")
            comp_source = (
                df.groupby("source")
                .agg(
                    Annonces=("prix", "count"),
                    Prix_Moyen=("prix", "mean"),
                    Prix_Median=("prix", "median"),
                    Km_Moyen=("kilometrage", "mean"),
                )
                .reset_index()
            )
            st.dataframe(
                comp_source,
                column_config={
                    "source": "Plateforme",
                    "Annonces": st.column_config.NumberColumn("Nb annonces", format="%d"),
                    "Prix_Moyen": st.column_config.NumberColumn(
                        "Prix moyen", format="%.0f €"
                    ),
                    "Prix_Median": st.column_config.NumberColumn(
                        "Prix médian", format="%.0f €"
                    ),
                    "Km_Moyen": st.column_config.NumberColumn(
                        "Km moyen", format="%.0f km"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("⛽ Prix médian par carburant")
            comp_energie = (
                df.groupby("energie")["prix"].median().sort_values(ascending=False)
            )
            st.bar_chart(comp_energie)

# ==============================================================================
# ONGLET 3 : ÉTAT DE LA BASE
# ==============================================================================
with tab_db:
    con_db = get_connection()
    df_sources = con_db.execute(
        """
        SELECT 
            source AS Plateforme,
            statut AS Statut,
            COUNT(*) AS Annonces,
            MIN(date_publication) AS Premiere_Collecte,
            MAX(date_derniere_vue) AS Derniere_Collecte
        FROM annonces
        GROUP BY source, statut
        ORDER BY Annonces DESC
        """
    ).df()

    df_top = con_db.execute(
        """
        SELECT marque AS Marque, COUNT(*) AS Vehicules
        FROM vehicules
        WHERE marque IS NOT NULL AND marque != 'INCONNU'
        GROUP BY marque
        ORDER BY Vehicules DESC
        LIMIT 30
        """
    ).df()
    con_db.close()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("📦 Volume par source et statut")
        st.dataframe(df_sources, use_container_width=True, hide_index=True)
    with col_d2:
        st.subheader("🏆 Top 30 des marques en base (par volume)")
        chart_top = (
            alt.Chart(df_top)
            .mark_bar(color="#2563eb")
            .encode(
                x=alt.X("Marque:N", sort="-y", title="Marque"),
                y=alt.Y("Vehicules:Q", title="Véhicules uniques"),
                tooltip=["Marque", "Vehicules"],
            )
        )
        st.altair_chart(chart_top, use_container_width=True)