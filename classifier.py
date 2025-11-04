"""
Module de classification pour prédire les titres de postes
Basé sur XGBoost et embeddings de compétences
"""

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
from pathlib import Path


class JobClassifier:
    """
    Classificateur de jobs basé sur les compétences
    Utilise XGBoost + Sentence Transformers
    """
    
    # Seuils de confiance
    CONFIDENCE_THRESHOLD = 0.35  # Seuil minimum de confiance
    MODERATE_CONFIDENCE = 0.50   # Seuil de confiance modérée
    
    def __init__(self, models_dir: str = "models"):
        # Convertir en chemin absolu
        self.models_dir = Path(models_dir).resolve()
        
        # Construire les chemins absolus
        self.model_path = self.models_dir / "model_xgboost.pkl"
        self.encoder_path = self.models_dir / "label_encoder.pkl"
        self.embeddings_path = self.models_dir / "job_embeddings_all.npy"
        
        # Vérifier que les fichiers existent
        print(f"📁 Recherche des modèles dans: {self.models_dir}")
        for path in [self.model_path, self.encoder_path, self.embeddings_path]:
            if not path.exists():
                raise FileNotFoundError(f"  Fichier introuvable: {path}")
            print(f"✅ Trouvé: {path.name}")
        
        self._load_models()
    
    def _load_models(self):
        """Charge tous les modèles nécessaires"""
        try:
            # Chargement du modèle XGBoost
            model_path = self.models_dir / "model_xgboost.pkl"
            self.model = joblib.load(model_path)
            print(f"✓ Modèle XGBoost chargé depuis {model_path}")
            
            # Chargement du label encoder
            le_path = self.models_dir / "label_encoder.pkl"
            self.label_encoder = joblib.load(le_path)
            print(f"✓ Label encoder chargé depuis {le_path}")
            
            # Chargement du modèle de sentence embedding
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Sentence Transformer chargé (all-MiniLM-L6-v2)")
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des modèles: {e}")
    
    def encode_skills(self, skills_text: str) -> np.ndarray:
        """
        Encode le texte de compétences en embedding
        
        Args:
            skills_text: Texte contenant les compétences (ex: "Python SQL Machine Learning")
        
        Returns:
            Embedding normalisé (vecteur numpy)
        """
        embedding = self.sentence_model.encode(
            [skills_text], 
            normalize_embeddings=True
        )
        return embedding
    
    def predict_job_title(self, skills_text: str, return_probabilities: bool = True,top_k: int = 3 ) -> Dict:
        """
        Prédit le titre de poste à partir des compétences
        
        Args:
            skills_text: Texte de compétences
            return_probabilities: Si True, retourne les probabilités pour chaque classe
            top_k: Nombre de classes les plus probables à retourner
        
        Returns:
            Dictionnaire contenant:
                - predicted_title: Titre prédit
                - confidence: Confiance de la prédiction (0-1)
                - is_confident: Si la confiance dépasse le seuil
                - warning: Message d'avertissement si nécessaire
                - top_predictions: Top-k des prédictions (si return_probabilities=True)
        """
        
        # Encodage des compétences
        skills_embedding = self.encode_skills(skills_text)
        
        # Prédiction avec probabilités
        probabilities = self.model.predict_proba(skills_embedding)[0]
        predicted_class = self.model.predict(skills_embedding)[0]
        
        # Décodage du titre
        predicted_title = self.label_encoder.inverse_transform([predicted_class])[0]
        max_probability = probabilities.max()
        
        # Vérification de la confiance
        is_confident = max_probability >= self.CONFIDENCE_THRESHOLD
        warning = None
        
        if max_probability < self.CONFIDENCE_THRESHOLD:
            warning = (
                f"Confiance insuffisante ({max_probability:.2%}). "
                "Les compétences saisies sont probablement hors du domaine data "
                "ou trop génériques/ambiguës."
            )
        elif max_probability < self.MODERATE_CONFIDENCE:
            warning = (
                f"Confiance modérée ({max_probability:.2%}). "
                "Les résultats peuvent être moins précis."
            )
        
        # Préparation du résultat
        result = {
            "predicted_title": predicted_title,
            "confidence": float(max_probability),
            "is_confident": is_confident,
            "warning": warning
        }
        
        # Ajout des top-k prédictions
        if return_probabilities:
            top_k_indices = probabilities.argsort()[::-1][:top_k]
            top_predictions = []
            
            for idx in top_k_indices:
                class_name = self.label_encoder.classes_[idx]
                prob = probabilities[idx]
                top_predictions.append({
                    "title": class_name,
                    "probability": float(prob)
                })
            
            result["top_predictions"] = top_predictions
            
            # Calcul de l'écart entre les 2 meilleures classes
            if len(top_predictions) >= 2:
                gap = top_predictions[0]["probability"] - top_predictions[1]["probability"]
                result["probability_gap"] = float(gap)
                
                if gap < 0.05:
                    result["ambiguity_warning"] = (
                        "Classes très proches : compétences ambiguës"
                    )
        
        return result
    
    def get_available_classes(self) -> List[str]:
        """
        Retourne la liste des classes (titres de postes) disponibles
        
        Returns:
            Liste des titres de postes
        """
        return self.label_encoder.classes_.tolist()
    
    def validate_skills(self, skills_text: str) -> Dict:
        """
        Valide que le texte de compétences est acceptable
        
        Args:
            skills_text: Texte à valider
        
        Returns:
            Dictionnaire avec:
                - is_valid: True si valide
                - error: Message d'erreur si invalide
        """
        if not skills_text or not skills_text.strip():
            return {
                "is_valid": False,
                "error": "Le texte de compétences ne peut pas être vide"
            }
        
        if len(skills_text.strip()) < 3:
            return {
                "is_valid": False,
                "error": "Le texte de compétences est trop court (minimum 3 caractères)"
            }
        
        return {"is_valid": True, "error": None}


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    # Initialisation du classificateur
    classifier = JobClassifier(models_dir="models")
    
    # Classes disponibles
    print("\n" + "="*70)
    print(" CLASSES DISPONIBLES")
    print("="*70)
    for i, classe in enumerate(classifier.get_available_classes(), 1):
        print(f"{i}. {classe}")
    
    # Test de prédiction
    print("\n" + "="*70)
    print(" TEST DE PRÉDICTION")
    print("="*70)
    
    test_skills = "Python SQL Machine Learning scikit-learn pandas"
    print(f"\n Compétences: {test_skills}")
    
    # Validation
    validation = classifier.validate_skills(test_skills)
    if not validation["is_valid"]:
        print(f" Erreur: {validation['error']}")
    else:
        # Prédiction
        result = classifier.predict_job_title(test_skills, top_k=3)
        
        print(f"\n Prédiction: {result['predicted_title']}")
        print(f" Confiance: {result['confidence']:.2%}")
        print(f" Confiant: {'Oui' if result['is_confident'] else 'Non'}")
        
        if result.get("warning"):
            print(f"  Avertissement: {result['warning']}")
        
        print("\n Top 3 des prédictions:")
        for i, pred in enumerate(result["top_predictions"], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {emoji} {pred['title']}: {pred['probability']:.2%}")
        
        if result.get("ambiguity_warning"):
            print(f"\n  {result['ambiguity_warning']}")
    
    print("\n" + "="*70)