"""
Module de recherche par similarité de jobs
Basé sur la similarité cosinus et embeddings
"""

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional
from pathlib import Path


class JobSimilaritySearch:
    """
    Recherche de jobs similaires par compétences
    Utilise XGBoost pour la classification + similarité cosinus
    """
    
    # Seuils (identiques à ton rech_sim4.py)
    CONFIDENCE_THRESHOLD = 0.35  # Seuil de confiance du modèle
    SIMILARITY_THRESHOLD = 0.30  # Seuil de similarité minimale
    MODERATE_CONFIDENCE = 0.50   # Seuil de confiance modérée
    MODERATE_SIMILARITY = 0.40   # Seuil de similarité modérée
    
    def __init__(self, models_dir= "models", data_dir = "data"):
        """
        Initialise le système de recherche
        
        Args:
            models_dir: Répertoire contenant les modèles
            data_dir: Répertoire contenant les données
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        self.xgboost_model = None
        self.label_encoder = None
        self.sentence_model = None
        self.df = None
        self.jobs_embeddings = None
        
        self._load_resources()
    
    def _load_resources(self):
        """Charge tous les modèles et données nécessaires"""
        try:
            # 1. Modèle XGBoost
            xgb_path = self.models_dir / "model_xgboost.pkl"
            self.xgboost_model = joblib.load(xgb_path)
            print(f"✓ XGBoost chargé depuis {xgb_path}")
            
            # 2. Label Encoder
            le_path = self.models_dir / "label_encoder.pkl"
            self.label_encoder = joblib.load(le_path)
            print(f"✓ Label encoder chargé depuis {le_path}")
            
            # 3. Sentence Transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence Transformer chargé")
            
            # 4. Base de données des jobs
            csv_path = self.data_dir / "BD_nettoyée.csv"
            self.df = pd.read_csv(csv_path)
            print(f"✓ Base de données chargée: {len(self.df)} jobs")
            
            # 5. Embeddings des jobs
            embeddings_path = self.models_dir / "job_embeddings_all.npy"
            self.jobs_embeddings = np.load(embeddings_path)
            print(f"✓ Embeddings chargés: {self.jobs_embeddings.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des ressources: {e}")
    
    def encode_skills(self, skills_text) :
        """
        Encode le texte de compétences en embedding
        
        Args:
            skills_text: Texte des compétences
        
        Returns:
            Embedding normalisé
        """
        embedding = self.sentence_model.encode(
            [skills_text], 
            normalize_embeddings=True
        )
        return embedding
    
    def recommend_jobs( self,  skills_text,  k = 5, include_scores = True
) :
        """
        Recommande des jobs basés sur les compétences
        
        Args:
            skills_text: Texte des compétences
            k: Nombre de jobs à recommander
            include_scores: Inclure les scores de similarité
        
        Returns:
            Dictionnaire contenant:
                - success: True si recommandations trouvées
                - predicted_class: Classe prédite
                - confidence: Confiance de la prédiction
                - similarity_max: Similarité maximale
                - similarity_avg: Similarité moyenne
                - recommendations: Liste des jobs recommandés
                - warnings: Liste d'avertissements
                - rejection_reason: Raison du rejet si applicable
        """
        
        # Encodage des compétences
        skills_embedding = self.encode_skills(skills_text)
        
        # ====================================================================
        # ÉTAPE 1 : PRÉDICTION DE LA CLASSE
        # ====================================================================
        probabilities = self.xgboost_model.predict_proba(skills_embedding)[0]
        max_proba = probabilities.max()
        predicted_class_num = self.xgboost_model.predict(skills_embedding)[0]
        predicted_class = self.label_encoder.inverse_transform([predicted_class_num])[0]
        
        # Top 3 des classes probables
        top3_indices = probabilities.argsort()[::-1][:3]
        top3_predictions = []
        for idx in top3_indices:
            top3_predictions.append({
                "class": self.label_encoder.classes_[idx],
                "probability": float(probabilities[idx])
            })
        
        # ====================================================================
        # SÉCURITÉ 1 : Vérification de la confiance
        # ====================================================================
        if max_proba < self.CONFIDENCE_THRESHOLD:
            # Calcul de l'écart entre les 2 meilleures classes
            top2_probas = probabilities[probabilities.argsort()[::-1][:2]]
            gap = top2_probas[0] - top2_probas[1]
            
            return {
                "success": False,
                "predicted_class": predicted_class,
                "confidence": float(max_proba),
                "top3_predictions": top3_predictions,
                "rejection_reason": "confidence_too_low",
                "message": (
                    f"Confiance insuffisante ({max_proba:.2%}). "
                    f"Le modèle hésite entre plusieurs classes. "
                    f"Écart entre les 2 meilleures: {gap:.2%}"
                ),
                "suggestions": [
                    "Ajoutez des compétences plus spécifiques",
                    "Essayez des termes techniques (Python, SQL, Machine Learning)",
                    "Vérifiez l'orthographe"
                ]
            }
        
        # ====================================================================
        # ÉTAPE 2 : RECHERCHE DES JOBS SIMILAIRES
        # ====================================================================
        # Filtrer les jobs de la classe prédite
        indices = self.df.index[self.df['title'] == predicted_class.lower().strip()].to_numpy()
        
        if len(indices) == 0:
            return {
                "success": False,
                "predicted_class": predicted_class,
                "confidence": float(max_proba),
                "rejection_reason": "no_jobs_in_class",
                "message": f"Aucun job trouvé pour la classe: {predicted_class}"
            }
        
        # Calcul des similarités
        skill_embedding_2d = skills_embedding.reshape(1, -1)
        similarities = cosine_similarity(
            skill_embedding_2d, 
            self.jobs_embeddings[indices]
        )[0]
        
        max_similarity = similarities.max()
        avg_similarity = similarities.mean()
        
        # ====================================================================
        # SÉCURITÉ 2 : Vérification de la similarité
        # ====================================================================
        if max_similarity < self.SIMILARITY_THRESHOLD:
            return {
                "success": False,
                "predicted_class": predicted_class,
                "confidence": float(max_proba),
                "similarity_max": float(max_similarity),
                "similarity_avg": float(avg_similarity),
                "top3_predictions": top3_predictions,
                "rejection_reason": "similarity_too_low",
                "message": (
                    f"Similarité trop faible ({max_similarity:.2%}). "
                    f"Même dans la classe '{predicted_class}', aucun job "
                    f"ne correspond vraiment aux compétences saisies."
                ),
                "suggestions": [
                    "Utilisez des compétences liées à la data/tech",
                    "Exemples: Python, SQL, Tableau, Machine Learning"
                ]
            }
        
        # ====================================================================
        # ÉTAPE 3 : GÉNÉRATION DES AVERTISSEMENTS
        # ====================================================================
        warnings = []
        
        if self.CONFIDENCE_THRESHOLD <= max_proba < self.MODERATE_CONFIDENCE:
            warnings.append({
                "type": "moderate_confidence",
                "message": f"Confiance modérée ({max_proba:.2%}). Les résultats peuvent être moins précis."
            })
        
        if self.SIMILARITY_THRESHOLD <= max_similarity < self.MODERATE_SIMILARITY:
            warnings.append({
                "type": "moderate_similarity",
                "message": f"Similarité modérée ({max_similarity:.2%}). Les jobs proposés correspondent partiellement."
            })
        
        # ====================================================================
        # ÉTAPE 4 : CRÉATION DES RECOMMANDATIONS
        # ====================================================================
        # Top-k indices triés par similarité
        top_k_indices_local = similarities.argsort()[::-1][:k]
        top_k_indices = indices[top_k_indices_local]
        top_k_similarities = similarities[top_k_indices_local]
        
        # Récupération des jobs
        recommendations = []
        for idx, (job_idx, sim_score) in enumerate(zip(top_k_indices, top_k_similarities), 1):
            job = self.df.iloc[job_idx]
            
            rec = {
                "rank": idx,
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "skills": eval(job['skills']) if isinstance(job['skills'], str) else job['skills']
            }
            
            if include_scores:
                rec["similarity_score"] = float(sim_score)
            
            recommendations.append(rec)
        
        # ====================================================================
        # ÉTAPE 5 : RÉSULTAT FINAL
        # ====================================================================
        result = {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": float(max_proba),
            "top3_predictions": top3_predictions,
            "similarity_max": float(max_similarity),
            "similarity_avg": float(avg_similarity),
            "num_jobs_in_class": len(indices),
            "recommendations": recommendations,
            "warnings": warnings
        }
        
        # Évaluation globale de la qualité
        if avg_similarity >= 0.50:
            result["quality"] = "excellent"
        elif avg_similarity >= 0.40:
            result["quality"] = "good"
        else:
            result["quality"] = "moderate"
        
        return result
    
    def get_job_details(self, job_index) :
        """
        Récupère les détails d'un job par son index
        
        Args:
            job_index: Index du job dans la base de données
        
        Returns:
            Dictionnaire avec les détails du job ou None
        """
        if job_index < 0 or job_index >= len(self.df):
            return None
        
        job = self.df.iloc[job_index]
        return {
            "title": job['title'],
            "company": job['company'],
            "location": job['location'],
            "skills": eval(job['skills']) if isinstance(job['skills'], str) else job['skills']
        }
    
    def get_stats(self):
        """
        Retourne des statistiques sur la base de données
        
        Returns:
            Statistiques générales
        """
        return {
            "total_jobs": len(self.df),
            "available_classes": self.label_encoder.classes_.tolist(),
            "jobs_per_class": self.df['title'].value_counts().to_dict()
        }


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    # Initialisation
    search_engine = JobSimilaritySearch(models_dir="models", data_dir="data")
    
    # Statistiques
    print("\n" + "="*70)
    print(" STATISTIQUES DE LA BASE")
    print("="*70)
    stats = search_engine.get_stats()
    print(f"Total de jobs: {stats['total_jobs']}")
    print(f"Classes disponibles: {', '.join(stats['available_classes'])}")
    
    # Test de recherche
    print("\n" + "="*70)
    print(" TEST DE RECHERCHE")
    print("="*70)
    
    test_skills = "Python SQL Machine Learning pandas scikit-learn"
    print(f"\n Compétences: {test_skills}")
    
    result = search_engine.recommend_jobs(test_skills, k=5, include_scores=True)
    
    if result["success"]:
        print(f"\n SUCCÈS")
        print(f" Classe prédite: {result['predicted_class']}")
        print(f" Confiance: {result['confidence']:.2%}")
        print(f" Similarité max: {result['similarity_max']:.2%}")
        print(f" Similarité moyenne: {result['similarity_avg']:.2%}")
        print(f" Qualité: {result['quality']}")
        
        # Avertissements
        if result["warnings"]:
            print(f"\n  AVERTISSEMENTS:")
            for warning in result["warnings"]:
                print(f"   • {warning['message']}")
        
        # Recommandations
        print(f"\n TOP {len(result['recommendations'])} RECOMMANDATIONS:")
        for rec in result["recommendations"]:
            emoji = "🥇" if rec['rank'] == 1 else "🥈" if rec['rank'] == 2 else "🥉" if rec['rank'] == 3 else "  "
            print(f"\n{emoji} #{rec['rank']} - {rec['title']}")
            print(f"    {rec['company']}")
            print(f"    {rec['location']}")
            print(f"    Similarité: {rec['similarity_score']:.2%}")
            print(f"    Compétences: {', '.join(rec['skills'][:5])}...")
    
    else:
        print(f"\n ÉCHEC")
        print(f"Raison: {result['rejection_reason']}")
        print(f"Message: {result['message']}")
        
        if result.get("suggestions"):
            print(f"\n SUGGESTIONS:")
            for suggestion in result["suggestions"]:
                print(f"   • {suggestion}")
    
    print("\n" + "="*70)
