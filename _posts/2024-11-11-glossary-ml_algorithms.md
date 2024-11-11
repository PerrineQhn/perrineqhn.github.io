---
title: 'Algorithmes Standards en Machine Learning'
date: 2024-11-11
permalink: /machine-learning-glossary/concepts/ml_algorithms/
toc: true
excerpt: "Concepts en apprentissage automatique : algorithmes standards de machine learning."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2024/11/glossary-ml_algorithms/
sidebar:
  title: "Glossaire ML"
  nav: sidebar-glossary
---

{% include base_path %}

## **Introduction**

Les algorithmes de machine learning se divisent en plusieurs grandes familles selon leur approche d'apprentissage. Chaque algorithme possède ses forces et faiblesses, et son choix dépend de la nature du problème à résoudre.

:bulb: <span class='intuition'>**Intuition :** Chaque algorithme peut être vu comme un outil différent dans une boîte à outils - un marteau est parfait pour planter un clou, mais inefficace pour visser une vis.</span>

## **1. Algorithmes Supervisés**

### Régression Linéaire

*La régression linéaire modélise la relation linéaire entre des variables d'entrée et une variable de sortie continue.*

#### Exemple : Prédiction de Prix Immobilier
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Données immobilières
data = {
    'surface_m2': [45, 60, 75, 90],
    'distance_centre': [2.5, 3.0, 4.5, 1.0],
    'prix': [200000, 280000, 310000, 420000]
}
df = pd.DataFrame(data)

# Préparation
X = df[['surface_m2', 'distance_centre']]
y = df['prix']
model = LinearRegression()
model.fit(X, y)

# Prédiction
nouveau_bien = [[70, 2.0]]
prix_predit = model.predict(nouveau_bien)
print(f"Prix prédit : {prix_predit[0]:,.0f}€")
```

:bulb: <span class='intuition'>**Intuition :** Imaginez une ligne qui tente de passer au plus près de tous les points d'un nuage de points.</span>

#### Formulation Mathématique
- Modèle : $$y = wx + b$$ 
- Fonction de coût : $$J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$$

#### Applications Clés
- Prédiction de prix immobiliers
- Estimation de ventes
- Prévision de consommation

#### Variantes
- Ridge (L2)
- Lasso (L1)
- Elastic Net

### Régression Logistique

*La régression logistique est un algorithme de classification qui estime la probabilité d'appartenance à une classe.*

#### Exemple : Détection de Spam
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Données d'emails
data = {
    'longueur_texte': [100, 2500, 30, 1500],
    'nb_liens': [1, 35, 2, 25],
    'est_spam': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Normalisation des features
scaler = StandardScaler()
X = scaler.fit_transform(df[['longueur_texte', 'nb_liens']])
y = df['est_spam']

# Modèle
model = LogisticRegression()
model.fit(X, y)

# Analyse nouvel email
nouvel_email = scaler.transform([[800, 12]])
proba_spam = model.predict_proba(nouvel_email)[0][1]
print(f"Probabilité de spam : {proba_spam:.2%}")
```

:bulb: <span class='intuition'>**Intuition :** C'est comme tracer une frontière qui sépare au mieux deux groupes de points.</span>

#### Formulation Mathématique
- Fonction sigmoïde : $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Probabilité : $$P(y=1|x) = \sigma(wx + b)$$

#### Applications Clés
- Détection de fraude
- Diagnostic médical
- Classification de spam

### Random Forest

*Ensemble d'arbres de décision combinant leurs prédictions.*

#### Exemple : Prédiction de Diabète
```python
from sklearn.ensemble import RandomForestClassifier

# Données patients
data = {
    'glucose': [85, 168, 122, 145],
    'imc': [22.1, 30.5, 24.3, 29.8],
    'age': [31, 45, 35, 50],
    'diabetique': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Modèle avec paramètres optimisés
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5
)
X = df[['glucose', 'imc', 'age']]
rf.fit(X, df['diabetique'])

# Importance des features
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
print("Importance des variables :")
print(importances.sort_values('importance', ascending=False))
```

:bulb: <span class='intuition'>**Intuition :** C'est comme demander l'avis à un groupe d'experts et prendre la décision majoritaire.</span>

#### Principes Clés
1. Bagging (échantillonnage avec remise)
2. Feature sampling
3. Vote majoritaire/moyenne

## **2. Algorithmes Non Supervisés**

### K-Means

*Algorithme de clustering qui regroupe les données en K clusters.*

#### Exemple : Segmentation Client
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Données clients
data = {
    'depense_moyenne': [100, 800, 200, 1200, 150, 900],
    'frequence_achat': [2, 12, 3, 15, 2, 10]
}
df = pd.DataFrame(data)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

# Visualisation
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['depense_moyenne'], 
                     df['frequence_achat'],
                     c=df['cluster'], 
                     cmap='viridis')
plt.xlabel('Dépense Moyenne')
plt.ylabel('Fréquence d\'Achat')
plt.title('Segments Clients')
plt.colorbar(scatter, label='Cluster')

# Analyse des centres
centres = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=['depense_moyenne', 'frequence_achat']
)
print("\nCentres des clusters :")
print(centres)
```

:bulb: <span class='intuition'>**Intuition :** Imaginez que vous devez regrouper des billes de différentes couleurs en tas, sans connaître les couleurs à l'avance.</span>

#### Algorithme
1. Initialisation aléatoire des centres
2. Attribution des points au centre le plus proche
3. Mise à jour des centres
4. Répétition jusqu'à convergence

#### Critères de Choix de K
- Méthode du coude
- Silhouette score
- Gap statistic

### PCA (Analyse en Composantes Principales)

*Technique de réduction de dimensionnalité préservant la variance maximale.*

#### Exemple : Réduction de Dimensionnalité d'Images
```python
from sklearn.decomposition import PCA
import numpy as np

# Création de données image simulées (28x28 pixels)
n_samples = 100
n_features = 28 * 28
X = np.random.rand(n_samples, n_features)

# Application PCA
pca = PCA(n_components=0.95)  # Garde 95% de la variance
X_reduit = pca.fit_transform(X)

# Analyse de la réduction
n_composantes = X_reduit.shape[1]
variance_ratio = pca.explained_variance_ratio_

print(f"Dimensions originales : {X.shape}")
print(f"Dimensions après PCA : {X_reduit.shape}")
print(f"Variance expliquée cumulée : {sum(variance_ratio):.2%}")

# Visualisation variance expliquée
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(variance_ratio))
plt.xlabel('Nombre de Composantes')
plt.ylabel('Variance Expliquée Cumulée')
plt.title('Scree Plot')
plt.grid(True)
```

:bulb: <span class='intuition'>**Intuition :** C'est comme prendre une photo d'un objet 3D selon l'angle qui montre le plus de détails possibles.</span>

#### Étapes
1. Centrage des données
2. Calcul de la matrice de covariance
3. Décomposition en valeurs propres
4. Projection sur les composantes principales

## **3. Métriques d'Évaluation**

### Classification

#### Exemple : Évaluation d'un Modèle de Diagnostic
```python
from sklearn.metrics import confusion_matrix, classification_report

# Résultats du modèle
y_true = [1, 0, 1, 1, 0, 0, 1]  # 1: Malade, 0: Sain
y_pred = [1, 0, 0, 1, 0, 1, 1]

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print("Matrice de confusion :")
print(cm)

# Métriques détaillées
print("\nRapport de classification :")
print(classification_report(y_true, y_pred))

# Calculs manuels
TP = cm[1,1]  # Vrais Positifs
TN = cm[0,0]  # Vrais Négatifs
FP = cm[0,1]  # Faux Positifs
FN = cm[1,0]  # Faux Négatifs

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"\nPrécision : {precision:.2f}")
print(f"Recall : {recall:.2f}")
print(f"F1-Score : {f1:.2f}")
```
- **Accuracy** : Proportion de prédictions correctes
- **Précision** : $$\frac{TP}{TP+FP}$$
- **Recall** : $$\frac{TP}{TP+FN}$$
- **F1-Score** : $$2 \times \frac{Précision \times Recall}{Précision + Recall}$$

### Régression

#### Exemple : Évaluation d'un Modèle de Prévision
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Données
y_true = [10, 15, 12, 18, 20, 16, 13, 21]
y_pred = [11, 15.5, 11.5, 17, 19, 16.5, 14, 20]

# Calcul des métriques
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² : {r2:.2f}")

# Analyse des résidus
residus = np.array(y_true) - np.array(y_pred)
print(f"\nRésidus moyens : {np.mean(residus):.2f}")
print(f"Écart-type résidus : {np.std(residus):.2f}")

# Visualisation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred)
plt.plot([min(y_true), max(y_true)], 
         [min(y_true), max(y_true)], 'r--')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')

plt.subplot(1, 2, 2)
plt.hist(residus, bins=10)
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Distribution des Résidus')
```

- **MSE** : $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE** : $$\sqrt{MSE}$$
- **MAE** : $$\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
- **R²** : $$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

## **4. Bonnes Pratiques**

### Préparation des Données
1. Nettoyage des valeurs manquantes
2. Normalisation/Standardisation
3. Encodage des variables catégorielles
4. Train/Test/Validation split

### Optimisation
- Grid search
- Random search
- Validation croisée
- Early stopping