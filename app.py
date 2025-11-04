"""
Interface Streamlit pour le système de recommandation de jobs
Consomme l'API FastAPI
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION DE L'API
# ============================================================================
API_URL = "http://localhost:8000"

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def check_api_health() -> bool:
    """Vérifie si l'API est accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False


def get_prediction(skills: str, top_k: int = 3) -> Dict[str, Any]:
    """Appelle l'endpoint /predict"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"skills": skills, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def get_recommendations(skills: str, k: int = 5, include_scores: bool = True) -> Dict[str, Any]:
    """Appelle l'endpoint /recommend"""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"skills": skills, "k": k, "include_scores": include_scores},
            timeout=10
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def get_stats() -> Dict[str, Any]:
    """Récupère les statistiques"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except:
        return {"success": False, "error": "Impossible de récupérer les statistiques"}


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================
def main():
    # Header
    st.title("🎯 Job Recommendation System")
    st.markdown("### Trouvez les jobs qui correspondent à vos compétences")
    st.markdown("---")
    
    # Vérification de l'API
    api_status = check_api_health()
    
    if not api_status:
        st.error(" **L'API n'est pas accessible !**")
        st.warning(f"Assurez-vous que l'API FastAPI est lancée sur `{API_URL}`")
        st.code("python main.py", language="bash")
        st.stop()
    
    st.success("✅ API connectée")
    
    # ===== 1. DÉFINITION DU MODE =====
    st.subheader("⚙️ Paramètres")
    
    col_mode, col_num = st.columns([2, 1])
    
    with col_mode:
        mode = st.radio(
            "Mode d'analyse",
            ["prediction", "recommendation"],
            format_func=lambda x: "🔮 Prédiction simple" if x == "prediction" else "🎯 Recommandation complète",
            horizontal=True,
            index=0
        )
    
    with col_num:
        if mode == "recommendation":
            num_recommendations = st.slider(
                "Nombre de recommandations",
                min_value=1,
                max_value=10,
                value=5
            )
        else:
            num_recommendations = 5
    
    st.markdown("---")
    
    # ===== 2. ZONE DE SAISIE =====
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Vos compétences")
        skills_input = st.text_area(
            "Entrez vos compétences (séparées par des espaces, virgules ou tirets)",
            placeholder="Exemple: Python SQL Machine Learning pandas scikit-learn",
            height=100
        )
    
    with col2:
        st.subheader("🚀 Actions")
        analyze_button = st.button(
            "✨ Analyser mes compétences",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ===== 3. TRAITEMENT =====
    if analyze_button and skills_input.strip():
        with st.spinner("🔄 Analyse en cours..."):
            
            if mode == "prediction":
                result = get_prediction(skills_input, top_k=3)
                
                if result["success"]:
                    data = result["data"]
                    
                    # Résultat principal
                    st.subheader("🎯 Résultat de la prédiction")
                    
                    col_result1, col_result2 = st.columns([2, 1])
                    
                    with col_result1:
                        st.markdown(f"### 🏆 **{data['predicted_title']}**")
                        
                        # Confiance
                        confidence = data['confidence']
                        confidence_color = "green" if confidence >= 0.5 else "orange" if confidence >= 0.35 else "red"
                        st.markdown(f"**Confiance:** :{confidence_color}[{confidence:.1%}]")
                        st.progress(confidence)
                        
                        if data.get("warning"):
                            st.warning(f"⚠️ {data['warning']}")
                        
                        if data.get("ambiguity_warning"):
                            st.info(f"ℹ️ {data['ambiguity_warning']}")
                    
                    with col_result2:
                        st.metric(
                            "Confiance",
                            f"{confidence:.1%}",
                            delta="Élevée" if confidence >= 0.5 else "Modérée" if confidence >= 0.35 else "Faible"
                        )
                        
                        if data.get("probability_gap"):
                            st.metric("Écart avec 2ème classe", f"{data['probability_gap']:.1%}")
                    
                    # Top prédictions
                    st.subheader("📊 Top 3 des prédictions")
                    top_preds = data['top_predictions']
                    df_preds = pd.DataFrame(top_preds)
                    
                    fig = px.bar(
                        df_preds,
                        x='probability',
                        y='title',
                        orientation='h',
                        color='probability',
                        color_continuous_scale='Blues',
                        text='probability'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig.update_layout(
                        xaxis_title="Probabilité",
                        yaxis_title="",
                        showlegend=False,
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f" Erreur: {result['error']}")
            
            elif mode == "recommendation":
                result = get_recommendations(skills_input, k=num_recommendations, include_scores=True)
                
                if result["success"]:
                    data = result["data"]
                    
                    if data["success"]:
                        # ===== EN-TÊTE DES RÉSULTATS (SANS CLASSE) =====
                        st.subheader("✅ Recommandations trouvées")
                        
                        # Affichage simplifié sans la classe prédite
                        col_header1, col_header2, col_header3 = st.columns(3)
                        
                        with col_header1:
                            st.metric("Confiance", f"{data['confidence']:.1%}")
                        
                        with col_header2:
                            if data.get("similarity_max"):
                                st.metric("Similarité max", f"{data['similarity_max']:.1%}")
                        
                        with col_header3:
                            if data.get("quality"):
                                quality_emoji = "🟢" if data["quality"] == "excellent" else "🟡" if data["quality"] == "good" else "🟠"
                                st.metric("Qualité", f"{quality_emoji} {data['quality'].title()}")
                        
                        # Message de qualité user-friendly
                        if data.get("quality_message"):
                            st.success(f"✨ {data['quality_message']}")
                        
                        # Avertissements
                        if data.get("warnings"):
                            for warning in data["warnings"]:
                                st.warning(f"⚠️ {warning['message']}")
                        
                        st.markdown("---")
                        
                        # ===== JOBS RECOMMANDÉS =====
                        st.subheader(f"🎯 Top {len(data['recommendations'])} des jobs recommandés")
                        
                        for rec in data["recommendations"]:
                            rank_emoji = "🥇" if rec['rank'] == 1 else "🥈" if rec['rank'] == 2 else "🥉" if rec['rank'] == 3 else "📌"
                            
                            with st.container():
                                col_job1, col_job2 = st.columns([3, 1])
                                
                                with col_job1:
                                    st.markdown(f"### {rank_emoji} **{rec['title'].upper()}**")
                                    st.markdown(f"**🏢 {rec['company']}**")
                                    st.markdown(f"📍 {rec['location']}")
                                    
                                    # Compétences
                                    skills_str = ", ".join(rec['skills'][:10])
                                    if len(rec['skills']) > 10:
                                        skills_str += f" ... (+{len(rec['skills']) - 10} autres)"
                                    
                                    st.markdown(f"**🔧 Compétences:** {skills_str}")
                                
                                with col_job2:
                                    if rec.get('similarity_score'):
                                        score = rec['similarity_score']
                                        st.metric(
                                            "Correspondance",
                                            f"{score:.1%}",
                                            delta="Excellente" if score >= 0.6 else "Bonne" if score >= 0.4 else "Modérée"
                                        )
                                        
                                        # Jauge circulaire
                                        fig = go.Figure(go.Indicator(
                                            mode="gauge+number",
                                            value=score * 100,
                                            domain={'x': [0, 1], 'y': [0, 1]},
                                            gauge={
                                                'axis': {'range': [None, 100]},
                                                'bar': {'color': "darkblue"},
                                                'steps': [
                                                    {'range': [0, 40], 'color': "lightgray"},
                                                    {'range': [40, 60], 'color': "lightyellow"},
                                                    {'range': [60, 100], 'color': "lightgreen"}
                                                ],
                                                'threshold': {
                                                    'line': {'color': "red", 'width': 4},
                                                    'thickness': 0.75,
                                                    'value': 90
                                                }
                                            }
                                        ))
                                        
                                        fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                st.markdown("---")
                        
                        # Résumé des similarités
                        if data.get("similarity_avg"):
                            st.info(f"📊 **Correspondance moyenne:** {data['similarity_avg']:.1%}")
                    
                    else:
                        # Rejet
                        st.error(f" {data['message']}")
                        
                        if data.get("suggestions"):
                            st.subheader("💡 Suggestions")
                            for suggestion in data["suggestions"]:
                                st.write(f"• {suggestion}")
                
                else:
                    st.error(f" Erreur: {result['error']}")
    
    elif analyze_button and not skills_input.strip():
        st.warning(" Veuillez entrer vos compétences avant d'analyser")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>🎯 Job Recommendation System | Powered by XGBoost & Sentence Transformers</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    main()