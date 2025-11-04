"""
Script de comparaison des modèles entraînés
Usage: python compare_models.py
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# CHARGEMENT DES RÉSULTATS
# ============================================================================
print("\n" + "="*70)
print(" COMPARAISON DES MODÈLES ENTRAÎNÉS")
print("="*70 + "\n")

model_keys = ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']
results_all = {}

for key in model_keys:
    results_file = f'results_{key}.pkl'
    if Path(results_file).exists():
        results_all[key] = joblib.load(results_file)
        print(f"✓ Chargé : {results_all[key]['model_name']}")
    else:
        print(f"  Manquant : {results_file}")

if not results_all:
    print("\n Aucun modèle trouvé ! Entraînez d'abord les modèles.")
    exit()

# ============================================================================
# TABLEAU DE COMPARAISON
# ============================================================================
print("\n" + "="*70)
print(" TABLEAU DES PERFORMANCES")
print("="*70 + "\n")

print(f"{'Modèle':<25} {'Score CV':<12} {'Score Test':<12} {'Temps (min)':<12}")
print("-" * 70)

best_model_key = None
best_score = 0

for key, results in results_all.items():
    model_name = results['model_name']
    cv_score = results['cv_score']
    test_score = results['test_score']
    training_time = results['training_time'] / 60
    
    print(f"{model_name:<25} {cv_score:.4f}       {test_score:.4f}       {training_time:.2f}")
    
    if test_score > best_score:
        best_score = test_score
        best_model_key = key

print("\n" + "="*70)
print(f"🏆 MEILLEUR MODÈLE : {results_all[best_model_key]['model_name']}")
print(f"   Accuracy : {best_score:.4f}")
print("="*70)

# ============================================================================
# GRAPHIQUE DE COMPARAISON DES PERFORMANCES
# ============================================================================
model_names = [results_all[key]['model_name'] for key in results_all.keys()]
cv_scores = [results_all[key]['cv_score'] for key in results_all.keys()]
test_scores = [results_all[key]['test_score'] for key in results_all.keys()]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, cv_scores, width, label='Score CV', color='skyblue')
bars2 = ax.bar(x + width/2, test_scores, width, label='Score Test', color='lightcoral')

ax.set_xlabel('Modèles', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Comparaison des performances des modèles', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Graphique sauvegardé : models_comparison.png")

# ============================================================================
# GRAPHIQUE DES TEMPS D'ENTRAÎNEMENT
# ============================================================================
training_times = [results_all[key]['training_time'] / 60 for key in results_all.keys()]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if i == best_model_key else 'steelblue' 
          for i, key in enumerate(results_all.keys())]
bars = ax.bar(model_names, training_times, color=colors)

ax.set_xlabel('Modèles', fontsize=12)
ax.set_ylabel('Temps d\'entraînement (minutes)', fontsize=12)
ax.set_title('Comparaison des temps d\'entraînement', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

# Ajouter les valeurs
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('training_times.png', dpi=300, bbox_inches='tight')
print(f"✓ Temps d'entraînement sauvegardés : training_times.png")

# ============================================================================
# AFFICHAGE DES MEILLEURS HYPERPARAMÈTRES
# ============================================================================
print("\n" + "="*70)
print(f" MEILLEURS HYPERPARAMÈTRES : {results_all[best_model_key]['model_name']}")
print("="*70 + "\n")

for param, value in results_all[best_model_key]['best_params'].items():
    print(f"   {param}: {value}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*70)
print(" RÉSUMÉ")
print("="*70)
print(f"\n  {len(results_all)} modèles comparés")
print(f"  Meilleur modèle : {results_all[best_model_key]['model_name']}")
print(f"  Accuracy test : {best_score:.4f}")
print(f"\n  Fichiers générés :")
print(f"   • models_comparison.png")
print(f"   • training_times.png")
print("\n" + "="*70 + "\n")