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

### Métriques de Classification

*Les métriques de classification évaluent la qualité des prédictions pour des variables catégorielles.*

:bulb: <span class='intuition'>**Intuition :** Ces métriques sont comme différentes façons d'évaluer un test médical - on veut savoir combien de malades on détecte correctement (vrais positifs) et combien de personnes saines on diagnostique par erreur (faux positifs).</span>

#### Matrice de Confusion

```
            │ Prédiction Positive │ Prédiction Négative
────────────┼────────────────────┼────────────────────
Réel Positif│  Vrais Positifs    │   Faux Négatifs
            │       (TP)         │       (FN)
────────────┼────────────────────┼────────────────────
Réel Négatif│  Faux Positifs    │   Vrais Négatifs
            │       (FP)         │       (TN)
```

#### Métriques de Base

##### Accuracy (Précision Globale)

- **Formule** : $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
- **Interprétation** : Proportion de prédictions correctes
- **Caractéristiques** :
  - Simple à comprendre
  - Entre 0 et 1 (meilleur = 1)
  - Trompeuse si classes déséquilibrées
- **Exemple** : Accuracy = 0.95 signifie 95% de prédictions correctes

##### Précision

- **Formule** : $Precision = \frac{TP}{TP + FP}$
- **Interprétation** : Proportion de vrais positifs parmi les prédictions positives
- **Caractéristiques** :
  - Mesure la qualité des prédictions positives
  - Importante quand les faux positifs sont coûteux
- **Exemple** : En diagnostic médical, proportion de vrais malades parmi les patients diagnostiqués positifs

##### Recall (Sensibilité, Rappel)

- **Formule** : $Recall = \frac{TP}{TP + FN}$
- **Interprétation** : Proportion de positifs correctement identifiés
- **Caractéristiques** :
  - Mesure la capacité à trouver tous les positifs
  - Importante quand manquer un positif est grave
- **Exemple** : En diagnostic, proportion de malades correctement détectés

##### F1-Score

- **Formule** : $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
- **Interprétation** : Moyenne harmonique de la précision et du recall
- **Caractéristiques** :
  - Combine précision et recall
  - Entre 0 et 1 (meilleur = 1)
  - Bon compromis pour classes déséquilibrées

#### Métriques Avancées

##### ROC (Receiver Operating Characteristic)

- **Définition** : Courbe TPR vs FPR pour différents seuils
  - TPR (True Positive Rate) = Recall
  - FPR (False Positive Rate) = $\frac{FP}{FP + TN}$
- **AUC-ROC** : Aire sous la courbe ROC
  - Entre 0.5 (aléatoire) et 1 (parfait)
  - Indépendant du seuil choisi
  - Mesure le pouvoir discriminant

##### Precision-Recall Curve

- **Définition** : Courbe Precision vs Recall
- **Utilisation** : 
  - Préférable à ROC pour classes très déséquilibrées
  - Montre le compromis précision/recall

#### Cas Particuliers

##### Classification Multi-classes

- **Macro-moyenne** : Moyenne simple sur toutes les classes
- **Micro-moyenne** : Pondération par fréquence des classes
- **Weighted-moyenne** : Pondération personnalisée par classe

##### Classification Multi-labels

- **Hamming Loss** : Proportion d'labels mal classés
- **Subset Accuracy** : Exactitude sur l'ensemble des labels

#### Exemple Pratique

```python
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Données exemple
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.3, 0.2, 0.9, 0.7, 0.1]

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print("Matrice de confusion :")
print(cm)

# Rapport détaillé
print("\nRapport de classification :")
print(classification_report(y_true, y_pred))

# Courbe ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nAUC-ROC : {roc_auc:.3f}")

# Courbe Precision-Recall
precision, recall, _ = precision_recall_curve(y_true, y_prob)
```

#### Choix des Métriques selon le Contexte

1. **Classes équilibrées**
   - Accuracy : bonne mesure générale
   - F1-score : pour plus de détails

2. **Classes déséquilibrées**
   - Precision/Recall : selon le coût des erreurs
   - AUC-PR : meilleure vue d'ensemble

3. **Coût des erreurs asymétrique**
   - Precision : si FP coûteux
   - Recall : si FN coûteux
   - Ajustement du seuil de décision

4. **Applications spécifiques**
   - **Médical** : Priorité recall (ne pas manquer de malades)
   - **Spam** : Priorité precision (ne pas bloquer de bons emails)
   - **Fraude** : Compromis selon coûts business

### Régression

### Métriques de Régression

*Les métriques de régression évaluent la qualité des prédictions pour des variables continues.*

:bulb: <span class='intuition'>**Intuition :** Ces métriques sont comme différentes façons de mesurer la distance entre vos prédictions et la réalité.</span>

#### Mean Squared Error (MSE)

- **Formule** : $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Interprétation** : Moyenne des erreurs au carré
- **Caractéristiques** :
  - Pénalise fortement les grandes erreurs
  - Toujours positive
  - Unité au carré (ex: euros²)
- **Utilisation** : 
  - Utile pour l'optimisation
  - Sensible aux outliers

#### Root Mean Squared Error (RMSE)

- **Formule** : $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} = \sqrt{MSE}$
- **Interprétation** : Écart-type des erreurs de prédiction
- **Caractéristiques** :
  - Même unité que la variable cible
  - Plus interprétable que MSE
- **Exemple** : RMSE = 5€ signifie que l'erreur "typique" est de 5€

#### Mean Absolute Error (MAE)

- **Formule** : $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **Interprétation** : Moyenne des valeurs absolues des erreurs
- **Caractéristiques** :
  - Plus robuste aux outliers que MSE/RMSE
  - Même unité que la variable cible
  - Erreur médiane si distribution exponentielle
- **Comparaison MAE vs RMSE** :
  - MAE < RMSE : présence d'erreurs importantes
  - MAE ≈ RMSE : erreurs uniformément distribuées

#### Coefficient de Détermination (R²)

- **Formule** : $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$
- **Interprétation** : Proportion de variance expliquée par le modèle
- **Caractéristiques** :
  - Varie entre -∞ et 1
  - R² = 1 : prédiction parfaite
  - R² = 0 : équivalent à prédire la moyenne
  - R² < 0 : pire que prédire la moyenne
- **Limites** :
  - Peut augmenter artificiellement avec le nombre de variables
  - Ne garantit pas la qualité des prédictions

#### Comparaison des Métriques

- **MSE/RMSE** : 
  - Avantage : Bonne mesure pour l'optimisation
  - Inconvénient : Sensible aux outliers
- **MAE** :
  - Avantage : Plus robuste aux outliers
  - Inconvénient : Gradient non continu en zéro
- **R²** :
  - Avantage : Facile à interpréter (pourcentage)
  - Inconvénient : Peut être trompeur dans certains cas

#### Exemple Pratique

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Vraies valeurs et prédictions
y_true = [100, 150, 120, 180, 200]
y_pred = [110, 155, 115, 170, 190]

# Calcul des métriques
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
```


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