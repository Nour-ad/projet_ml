"""
Script d'entraînement modulaire pour un modèle spécifique
Usage: python train_model.py --model logistic
Options: logistic, random_forest, gradient_boosting, svm
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 50  # Nombre d'itérations pour RandomSearch

# PCA : mettre None pour désactiver, ou un nombre (ex: 50, 100, 200)
USE_PCA = None 
# Si USE_PCA = None, le script utilisera toutes les dimensions

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def normalize_title(title):
    """Normalise les titres de poste"""
    title_lower = str(title).lower()
    
    if 'data analyst' in title_lower or 'business analyst' in title_lower:
        return 'Data Analyst'
    elif 'data engineer' in title_lower:
        return 'Data Engineer'
    elif 'data scientist' in title_lower or 'machine learning' in title_lower:
        return 'Data Scientist'
    elif 'analyst' in title_lower:
        return 'Data Analyst'
    elif 'engineer' in title_lower:
        return 'Data Engineer'
    elif 'scientist' in title_lower:
        return 'Data Scientist'
    else:
        return 'Other'

def load_and_prepare_data():
    """Charge et prépare les données"""
    print("\n" + "="*70)
    print(" CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # Chargement
    df = pd.read_csv("BD_nettoyée.csv")
    X = np.load('skills_embeddings_all.npy')
    
    print(f"✓ Dimensions des embeddings : {X.shape}")
    
    # Normalisation des titres
    df['title_normalized'] = df['title'].apply(normalize_title)
    y = df['title_normalized'].values
    
    # Filtrage des classes rares
    counts = df['title_normalized'].value_counts()
    common_titles = counts[counts >= 15].index
    mask = df['title_normalized'].isin(common_titles)
    
    X = X[mask]
    y = y[mask]

    # Supprimer la classe "Other" (trop peu d'exemples et peu cohérente)
    mask_no_other = y != 'Other'
    X = X[mask_no_other]
    y = y[mask_no_other]
    print(f"✓ Classe 'Other' supprimée : {len(y)} exemples restants")
    
    print(f"✓ Classes conservées : {list(common_titles)}")
    print(f"✓ Nombre d'exemples : {len(y)}")
    
    # Encodage
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"✓ Train set : {len(X_train)} exemples")
    print(f"✓ Test set  : {len(X_test)} exemples")
    
    return X_train, X_test, y_train, y_test, le

def create_preprocessing_pipeline(use_pca=None):
    """Crée le pipeline de prétraitement (StandardScaler + PCA optionnel)"""
    steps = [('scaler', StandardScaler())]
    
    if use_pca is not None:
        steps.append(('pca', PCA(n_components=use_pca, random_state=RANDOM_STATE)))
        print(f"✓ PCA activé : réduction à {use_pca} dimensions")
    else:
        print(f"✓ PCA désactivé : utilisation de toutes les dimensions")
    
    return Pipeline(steps)

# ============================================================================
# CONFIGURATION DES MODÈLES
# ============================================================================
MODEL_CONFIGS = {
    'logistic': {
        'name': 'Logistic Regression',
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2', None],
            'model__solver': ['lbfgs', 'saga'],
            'model__class_weight': ['balanced', None]
        }
    },
    'random_forest': {
        'name': 'Random Forest',
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'params': {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': [10, 15, 20, 30, None],
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 8),
            'model__max_features': ['sqrt', 'log2', None],
            'model__class_weight': ['balanced', None]
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'model__n_estimators': randint(100, 400),
            'model__learning_rate': uniform(0.01, 0.19),
            'model__max_depth': randint(3, 15),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 8),
            'model__subsample': uniform(0.7, 0.3),
            'model__max_features': ['sqrt', 'log2', None]
        }
    },
    'xgboost': {
    'name': 'XGBoost',
    'model': XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,              # utilisation multithread CPU
        use_label_encoder=False,
        eval_metric='mlogloss'  # évite les warnings
    ),
    'params': {
        'model__n_estimators': randint(100, 400),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth': randint(3, 15),
        'model__subsample': uniform(0.7, 0.3),
        'model__colsample_bytree': uniform(0.7, 0.3),
        'model__gamma': uniform(0, 0.5),
        'model__min_child_weight': randint(1, 8)
    }
}

}

# ============================================================================
# FONCTION D'ENTRAÎNEMENT
# ============================================================================
def train_model(model_key, X_train, X_test, y_train, y_test, le):
    """Entraîne un modèle spécifique avec optimisation des hyperparamètres"""
    
    config = MODEL_CONFIGS[model_key]
    model_name = config['name']
    
    print("\n" + "="*70)
    print(f" ENTRAÎNEMENT : {model_name}")
    print("="*70)
    
    start_time = time.time()
    
    # Créer le pipeline complet (prétraitement + modèle)
    preprocessing = create_preprocessing_pipeline(use_pca=USE_PCA)
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', config['model'])
    ])
    
    # Random Search
    print(f"\n🔍 Recherche des meilleurs hyperparamètres...")
    print(f"   - Itérations : {N_ITER}")
    print(f"   - Cross-validation : {CV_FOLDS} folds")
    
    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=config['params'],
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # Entraînement
    random_search.fit(X_train, y_train)
    
    # Meilleurs paramètres
    print(f"\n✓ Meilleurs paramètres trouvés :")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Évaluation
    best_pipeline = random_search.best_estimator_
    cv_score = random_search.best_score_
    y_pred = best_pipeline.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"\n RÉSULTATS :")
    print(f"   Score CV (train) : {cv_score:.4f}")
    print(f"   Score Test       : {test_score:.4f}")
    print(f"   Temps d'entraînement : {training_time/60:.2f} minutes")
    
    # Rapport de classification
    print(f"\n Rapport de classification :")
    print(classification_report(
        y_test, y_pred,
        labels=np.unique(y_test),
        target_names=le.inverse_transform(np.unique(y_test)),
        zero_division=0
    ))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    class_names = le.inverse_transform(np.unique(y_test))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_file = f'confusion_matrix_{model_key}.png'
    plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Matrice sauvegardée : {confusion_file}")
    plt.close()
    
    # Sauvegarde du modèle
    model_file = f'model_{model_key}.pkl'
    results_file = f'results_{model_key}.pkl'
    
    joblib.dump(best_pipeline, model_file)
    
    results = {
        'model_name': model_name,
        'cv_score': cv_score,
        'test_score': test_score,
        'training_time': training_time,
        'best_params': random_search.best_params_,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    joblib.dump(results, results_file)
    
    print(f"✓ Modèle sauvegardé : {model_file}")
    print(f"✓ Résultats sauvegardés : {results_file}")
    
    return best_pipeline, results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraîner un modèle spécifique')
    parser.add_argument('--model', type=str, required=True,
                      choices=['logistic', 'random_forest', 'gradient_boosting', 'xgboost'],
                      help='Modèle à entraîner')
    
    args = parser.parse_args()
    
    print("\n" + "🚀 " + "="*66 + " 🚀")
    print(f"   ENTRAÎNEMENT DU MODÈLE : {MODEL_CONFIGS[args.model]['name']}")
    print("🚀 " + "="*66 + " 🚀")
    
    # Chargement des données
    X_train, X_test, y_train, y_test, le = load_and_prepare_data()
    
    # Sauvegarde du label encoder (une seule fois)
    joblib.dump(le, 'label_encoder.pkl')
    print("\n✓ Label encoder sauvegardé : label_encoder.pkl")
    
    # Entraînement
    model, results = train_model(args.model, X_train, X_test, y_train, y_test, le)
    
    print("\n" + "="*70)
    print(" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("="*70)
    print(f"\n  Fichiers générés :")
    print(f"   • model_{args.model}.pkl")
    print(f"   • results_{args.model}.pkl")
    print(f"   • confusion_matrix_{args.model}.png")
    print(f"   • label_encoder.pkl")
    print(f"\n  Accuracy finale : {results['test_score']:.4f}")
    print("="*70 + "\n")