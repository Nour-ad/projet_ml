"""
API FastAPI pour le système de recommandation de jobs
Endpoints: /predict (classification) et /recommend (similarité)
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
from fastapi.responses import JSONResponse

# Import des modules custom
sys.path.append(str(Path(__file__).parent))
from models.classifier import JobClassifier
from models.similarity import JobSimilaritySearch

# Définir les chemins absolus
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================================
# CONFIGURATION DE L'API
# ============================================================================
app = FastAPI(
    title="Job Recommendation API",
    description="API de recommandation de jobs basée sur les compétences",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (pour permettre les appels depuis un frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CHARGEMENT DES MODÈLES AU DÉMARRAGE
# ============================================================================
classifier = None
similarity_engine = None

@app.on_event("startup")
async def startup_event():
    """Charge les modèles au démarrage de l'API"""
    global classifier, similarity_engine
    
    print("\n" + "="*70)
    print(" 🚀 DÉMARRAGE DE L'API")
    print("="*70)
    
    try:
        # Chargement du classificateur
        print("\n Chargement du classificateur...")
        classifier = JobClassifier(models_dir=str(MODELS_DIR))
        
        # Chargement du moteur de similarité
        print("\n Chargement du moteur de similarité...")
        similarity_engine = JobSimilaritySearch(
        models_dir=str(MODELS_DIR),  # ✅
        data_dir=str(DATA_DIR)  # ✅
    )
        
        print("\n API prête !")
        print(" Documentation disponible sur: http://localhost:8000/docs")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n ERREUR lors du chargement: {e}")
        print("Vérifiez que tous les fichiers nécessaires sont présents.\n")
        raise

# ============================================================================
# MODÈLES PYDANTIC (SCHEMAS)
# ============================================================================

class SkillsRequest(BaseModel):
    """Requête avec compétences"""
    skills: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Compétences séparées par des espaces, virgules ou tirets",
        example="Python SQL Machine Learning pandas scikit-learn"
    )
    
    @validator('skills')
    def validate_skills(cls, v):
        """Valide que les compétences ne sont pas vides après strip"""
        if not v.strip():
            raise ValueError("Les compétences ne peuvent pas être vides")
        return v.strip()


class PredictionRequest(SkillsRequest):
    """Requête de prédiction"""
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de prédictions à retourner"
    )


class RecommendationRequest(SkillsRequest):
    """Requête de recommandation"""
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de jobs à recommander"
    )
    include_scores: bool = Field(
        default=True,
        description="Inclure les scores de similarité"
    )


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    predicted_title: str
    confidence: float
    is_confident: bool
    warning: Optional[str] = None
    top_predictions: List[Dict[str, Any]]
    probability_gap: Optional[float] = None
    ambiguity_warning: Optional[str] = None


class JobRecommendation(BaseModel):
    """Un job recommandé"""
    rank: int
    title: str
    company: str
    location: str
    skills: List[str]
    similarity_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Réponse de recommandation"""
    success: bool
    predicted_class: str
    confidence: float
    similarity_max: Optional[float] = None
    similarity_avg: Optional[float] = None
    quality: Optional[str] = None
    recommendations: Optional[List[JobRecommendation]] = None
    warnings: Optional[List[Dict[str, str]]] = None
    rejection_reason: Optional[str] = None
    message: Optional[str] = None
    suggestions: Optional[List[str]] = None
    top3_predictions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Réponse de santé de l'API"""
    status: str
    version: str
    models_loaded: bool


class StatsResponse(BaseModel):
    """Statistiques de la base"""
    total_jobs: int
    available_classes: List[str]
    jobs_per_class: Dict[str, int]

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Endpoint racine"""
    return {
        "message": "Job Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "recommend": "/recommend",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifie l'état de santé de l'API"""
    models_loaded = classifier is not None and similarity_engine is not None
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "version": "1.0.0",
        "models_loaded": models_loaded
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Classification"])
async def predict_job_title(request: PredictionRequest):
    """
    Prédit le titre de poste à partir des compétences
    
    - **skills**: Compétences (ex: "Python SQL Machine Learning")
    - **top_k**: Nombre de prédictions à retourner (défaut: 3)
    
    Retourne la classe prédite avec le niveau de confiance
    """
    
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le classificateur n'est pas chargé"
        )
    
    try:
        # Validation
        validation = classifier.validate_skills(request.skills)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation["error"]
            )
        
        # Prédiction
        result = classifier.predict_job_title(
            skills_text=request.skills,
            return_probabilities=True,
            top_k=request.top_k
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
async def recommend_jobs(request: RecommendationRequest):
    """
    Recommande des jobs similaires basés sur les compétences
    
    - **skills**: Compétences (ex: "Python SQL Machine Learning")
    - **k**: Nombre de jobs à recommander (défaut: 5)
    - **include_scores**: Inclure les scores de similarité (défaut: true)
    
    Retourne les jobs les plus similaires avec leurs détails
    """
    
    if similarity_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le moteur de similarité n'est pas chargé"
        )
    
    try:
        # Recommandation
        result = similarity_engine.recommend_jobs(
            skills_text=request.skills,
            k=request.k,
            include_scores=request.include_scores
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la recommandation: {str(e)}"
        )


@app.get("/stats",include_in_schema=False, response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Retourne les statistiques de la base de données
    
    - Nombre total de jobs
    - Classes disponibles
    - Répartition des jobs par classe
    """
    
    if similarity_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le moteur de similarité n'est pas chargé"
        )
    
    try:
        stats = similarity_engine.get_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )


@app.get("/classes", include_in_schema=False, tags=["Statistics"])
async def get_available_classes():
    """
    Retourne la liste des classes (titres de postes) disponibles
    """
    
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le classificateur n'est pas chargé"
        )
    
    try:
        classes = classifier.get_available_classes()
        return {
            "classes": classes,
            "count": len(classes)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des classes: {str(e)}"
        )

# ============================================================================
# GESTION DES ERREURS
# ============================================================================


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Gestionnaire d'erreur 404"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint non trouvé",
            "message": "Consultez la documentation sur /docs"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Gestionnaire d'erreur 500"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "message": "Contactez l'administrateur si le problème persiste"
        }
    )

# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "🚀 " + "="*66 + " 🚀")
    print("   DÉMARRAGE DU SERVEUR FASTAPI")
    print("🚀 " + "="*66 + " 🚀\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en développement
        log_level="info"
    )