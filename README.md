#  Système de Recommandation de Carrières basé sur les Compétences

Un système intelligent de classification et de recommandation de postes dans le domaine de la data, utilisant Machine Learning et recherche par similarité sémantique.


---

## **Table des matières**

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [API Endpoints](#api-endpoints)
- [Technologies](#technologies)

---

## **Vue d'ensemble**

Ce projet implémente un système complet de recommandation de carrières qui :
1. **Classifie** automatiquement un profil en 3 catégories : **Data Analyst**, **Data Engineer**, **Data Scientist**
2. **Recommande** les k meilleurs jobs correspondant aux compétences saisies
3. **Évalue** la confiance et la qualité des recommandations avec des seuils de sécurité

**Cas d'usage** : Orientation professionnelle, matching CV-offres d'emploi, analyse de profils tech.

---

## **Fonctionnalités**

### **Classification intelligente**
- Prédiction du métier cible basée sur les compétences
- Système de confiance à 3 niveaux (faible/modéré/élevé)
- Top-3 des carrières possibles avec probabilités

### **Recommandation par similarité** 
- Filtrage intelligent par classe prédite
- Calcul de similarité cosinus via embeddings (SentenceTransformer)
- Double seuil de sécurité (confiance + similarité)
- Scoring de qualité globale (excellent/good/moderate)

### **Robustesse**
- Validation des entrées (anti-erreurs)
- Gestion des cas limites avec suggestions
- Rejet des compétences hors domaine
- Warnings explicites pour profils incertains

### **Déploiement**
- **API REST** (FastAPI) pour intégration
- **Interface web** (Streamlit) interactive
- Latence optimisée (~55ms par requête)
- Architecture modulaire et scalable

---

## **Architecture**
```
┌─────────────────────────────────────────────────────────┐
│                   USER INPUT                            │
│           "Python, SQL, Machine Learning"               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              JobClassifier (classifier.py)              │
│  • Validation entrée                                    │
│  • Encoding → SentenceTransformer (384-dim)             │
│  • Prédiction → XGBoost Pipeline                        │
│  • Output : classe + confiance + top-3                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│         JobSimilaritySearch (similarity.py)             │
│  • Filtrage jobs par classe prédite                     │
│  • Similarité cosinus (embeddings)                      │
│  • Double seuil sécuritaire                             │
│  • Output : top-K jobs + scores + warnings              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    API / FRONTEND                       │
│  • FastAPI (main.py) : Endpoints REST                   │
│  • Streamlit (app.py) : Interface web                   │
└─────────────────────────────────────────────────────────┘
```

---

##  **Installation**

### **Prérequis**
- Python 3.8+
- pip

### **Étapes**

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/job-recommendation-system.git
cd job-recommendation-system
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger les données et modèles** (si non inclus)
```bash
# Placer les fichiers suivants :
# - data/BD_nettoyée.csv
# - models/model_xgboost.pkl
# - models/label_encoder.pkl
# - models/job_embeddings_all.npy
```

---

## **Utilisation**

### **1. Préparer les données**
```bash
python clean_db.py
```
**Output** : `BD_nettoyée.csv` (base de données nettoyée)

### **2. Entraîner les modèles**
```bash
# Entraîner un modèle spécifique
python train_model.py --model xgboost

# Options disponibles : logistic, random_forest, gradient_boosting, xgboost
```
**Output** : 
- `model_xgboost.pkl`
- `results_xgboost.pkl`
- `confusion_matrix_xgboost.png`
- `label_encoder.pkl`

### **3. Comparer les modèles**
```bash
python compare_models.py
```
**Output** : 
- `models_comparison.png`
- `training_times.png`
- Affichage du meilleur modèle

### **4. Tester le classificateur**
```bash
python classifier.py
```

### **5. Tester la recherche par similarité**
```bash
python similarity.py
```

### **6. Lancer l'API REST**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
L'API sera disponible sur `http://localhost:8000`  
Documentation interactive : `http://localhost:8000/docs`

### **7. Lancer l'interface web**
```bash
streamlit run app.py
```
L'interface sera disponible sur `http://localhost:8501`

---

## **Structure du projet**
```
job-recommendation-system/
│
├── data/
│   ├── clean_jobs.csv              # Dataset brut
│   └── BD_nettoyée.csv             # Dataset nettoyé
│
├── models/
│   ├── model_xgboost.pkl           # Modèle XGBoost entraîné
│   ├── label_encoder.pkl           # Encodeur de labels
│   ├── job_embeddings_all.npy      # Embeddings de jobs pré-calculés
│   ├── skills_embeddings_all.npy   # Embeddings de skills pré-calculés
│   
│
├── clean_db.py                     # Nettoyage de données
├── train_model.py                  # Entraînement de modèles
├── compare_models.py               # Comparaison de modèles
├── classifier.py                   # Module de classification
├── similarity.py                   # Module de recherche
├── main.py                         # API FastAPI
├── app.py                          # Interface Streamlit
│
├── requirements.txt                # Dépendances
└── README.md                       # Documentation
```


---

## **API Endpoints**

### **POST /predict**
Classifie un profil en fonction des compétences.

**Request Body:**
```json
{
  "skills_text": "Python SQL Machine Learning pandas scikit-learn"
}
```

**Response:**
```json
{
  "predicted_title": "Data Scientist",
  "confidence": 0.78,
  "is_confident": true,
  "warning": null,
  "top_predictions": [
    {"title": "Data Scientist", "probability": 0.78},
    {"title": "Data Analyst", "probability": 0.15},
    {"title": "Data Engineer", "probability": 0.07}
  ]
}
```

### **POST /recommend**
Recommande les meilleurs jobs correspondants.

**Request Body:**
```json
{
  "skills_text": "Python SQL TensorFlow",
  "k": 5
}
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "Data Scientist",
  "confidence": 0.85,
  "similarity_max": 0.72,
  "similarity_avg": 0.65,
  "quality": "good",
  "recommendations": [
    {
      "rank": 1,
      "title": "machine learning engineer",
      "company": "Tech Corp",
      "location": "Paris",
      "skills": ["python", "tensorflow", "ml"],
      "similarity_score": 0.72
    }
  ]
}
```

### **GET /stats**
Retourne les statistiques de la base de données.

**Response:**
```json
{
  "total_jobs": 5234,
  "available_classes": ["Data Analyst", "Data Engineer", "Data Scientist"],
  "jobs_per_class": {
    "data analyst": 1823,
    "data engineer": 1654,
    "data scientist": 1757
  }
}
```

---

## **Technologies**

| Catégorie | Technologies |
|-----------|--------------|
| **ML/DL** | XGBoost, scikit-learn, SentenceTransformer |
| **NLP** | spaCy, wordninja, all-MiniLM-L6-v2 |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit, Plotly |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Serialization** | Joblib |
