---
title: 'Algorithmes et Métriques Standards d'Apprentissage Automatique'
date: 2024-10-08
#modified: 
permalink: /machine-learning-glossary/concepts/ml-algorithms
toc: false
excerpt: "Concepts en apprentissage automatique : algorithmes et métriques standards."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2024/11/glossary-ml-algorithms
sidebar:
  title: "Glossaire ML"
  nav: sidebar-glossary

---
{% include base_path %}

## **1. Algorithmes Supervisés**

### 1.1 Régression Linéaire

#### Exemple Concret : Prédiction du Prix Immobilier
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Données
data = {
    'surface_m2': [45, 60, 75, 90, 120, 150],
    'nb_chambres': [1, 2, 2, 3, 3, 4],
    'distance_centre': [2.5, 3.0, 4.5, 1.0, 5.0, 3.5],
    'prix': [200000, 280000, 310000, 420000, 450000, 590000]
}
df = pd.DataFrame(data)

# Préparation
X = df[['surface_m2', 'nb_chambres', 'distance_centre']]
y = df['prix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Interprétation
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("Coefficients :")
print(coefficients)
print(f"R² Score : {model.score(X_test, y_test):.3f}")

# Prédiction pour un nouvel appartement
nouveau = pd.DataFrame([[80, 2, 3.0]], columns=X.columns)
prix_predit = model.predict(nouveau)[0]
print(f"Prix prédit : {prix_predit:.2f}€")
```

#### Interprétation :
- Un coefficient de 2000 pour surface_m2 signifie que chaque m² supplémentaire augmente le prix de 2000€
- R² de 0.85 indique que 85% de la variance des prix est expliquée par le modèle
- **Description** : 
  - Modélise $$y = wx + b$$ où w est le vecteur de poids et b le biais
  - Minimise la somme des carrés des résidus (MSE)
- **Formule du coût** : $$J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$$
- **Applications détaillées** :
  - Prédiction des prix immobiliers basée sur la surface, l'emplacement, etc.
  - Estimation des ventes en fonction des dépenses marketing
  - Prévision de la consommation énergétique
- **Variantes** :
  - Ridge (L2) : ajoute $$\alpha\sum w^2$$ au coût
  - Lasso (L1) : ajoute $$\alpha\sum |w|$$ au coût
  - Elastic Net : combine L1 et L2
- **Implémentation (exemple Python)** :
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 1.2 Régression Logistique

#### Exemple Concret : Prédiction de Fraude Bancaire
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Données
data = {
    'montant': [100, 2500, 30, 15000, 450, 3000],
    'heure': [14, 23, 15, 2, 11, 22],
    'pays_habituel': [1, 1, 1, 0, 1, 0],
    'fraude': [0, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Préparation
X = df[['montant', 'heure', 'pays_habituel']]
y = df['fraude']

# Normalisation importante pour la régression logistique
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Évaluation
y_pred = model.predict(X_scaled)
print(classification_report(y, y_pred))

# Probabilités pour une nouvelle transaction
nouvelle_transaction = pd.DataFrame([[500, 13, 1]], columns=X.columns)
nouvelle_transaction_scaled = scaler.transform(nouvelle_transaction)
proba_fraude = model.predict_proba(nouvelle_transaction_scaled)[0][1]
print(f"Probabilité de fraude : {proba_fraude:.2%}")

# Seuil de décision personnalisé
seuil = 0.75
prediction = "Frauduleuse" if proba_fraude > seuil else "Légitime"
print(f"Classification avec seuil {seuil}: {prediction}")
```

#### Points Clés :
- Normalisation cruciale pour la performance
- Ajustement possible du seuil selon le coût des faux positifs/négatifs
- Interprétation des coefficients pour l'importance des features
- **Description** :
  - Utilise la fonction sigmoïde : $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  - Probabilité de classe : $$P(y=1|x) = \sigma(wx + b)$$
- **Fonction de coût** : 
  $$J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(h(x_i)) + (1-y_i)\log(1-h(x_i))]$$
- **Hyperparamètres clés** :
  - C : inverse de la régularisation (plus C est grand, moins de régularisation)
  - max_iter : nombre maximum d'itérations
  - solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}
- **Applications détaillées** :
  - Prédiction de la probabilité de défaut de paiement
  - Classification d'emails (spam/non-spam)
  - Diagnostic médical (malade/sain)
- **Extension multi-classe** :
  - One-vs-Rest : un classifieur par classe
  - Softmax : généralisation multi-classe directe

### 1.3 Arbres de Décision
- **Description détaillée** :
  - Structure hiérarchique de nœuds de décision
  - Chaque nœud divise les données selon un critère
- **Critères de division** :
  - Gini : $$1 - \sum_{i=1}^{c} (p_i)^2$$
  - Entropie : $$-\sum_{i=1}^{c} p_i \log_2(p_i)$$
  - MSE (régression) : $$\frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$
- **Hyperparamètres importants** :
  - max_depth : profondeur maximale
  - min_samples_split : nombre minimum d'échantillons pour diviser
  - min_samples_leaf : nombre minimum d'échantillons par feuille
  - max_features : nombre maximum de features à considérer
- **Techniques d'élagage** :
  - Pré-élagage : limites sur la croissance
  - Post-élagage : réduction après construction
- **Exemple d'implémentation** :
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2
)
```

### 1.4 Random Forest
- **Principe détaillé** :
  - Bagging (Bootstrap Aggregating)
  - Feature sampling à chaque split
  - Vote majoritaire ou moyenne des prédictions
- **Paramètres clés** :
  - n_estimators : nombre d'arbres
  - max_features : $$\sqrt{n}$$ pour classification, $$n/3$$ pour régression
  - bootstrap : True pour bagging
- **Importance des variables** :
  - Basée sur la diminution moyenne de l'impureté
  - Permutation importance
- **Out-of-Bag Score** :
  - Validation naturelle sur échantillons non utilisés
  - ≈ 37% des données pour chaque arbre

### 1.5 Support Vector Machines (SVM)
- **Formulation mathématique** :
  - Problème primal : $$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$
  - Contraintes : $$y_i(w^Tx_i + b) \geq 1 - \xi_i$$
- **Kernels courants** :
  - Linéaire : $$K(x,y) = x^Ty$$
  - RBF : $$K(x,y) = \exp(-\gamma||x-y||^2)$$
  - Polynomial : $$K(x,y) = (x^Ty + c)^d$$
- **Hyperparamètres critiques** :
  - C : compromis régularisation/erreur
  - gamma : influence d'un point d'entraînement
  - kernel : type de transformation

## **2. Algorithmes Non Supervisés**

### 2.1 K-Means
- **Algorithme détaillé** :
  1. Initialisation aléatoire des centroïdes
  2. Attribution des points au centroïde le plus proche
  3. Mise à jour des centroïdes
  4. Répétition jusqu'à convergence
- **Méthodes d'initialisation** :
  - K-means++ : initialisation intelligente
  - Random : points aléatoires
- **Critères d'arrêt** :
  - Nombre max d'itérations
  - Changement minimal des centroïdes
  - Convergence de l'inertie
- **Choix optimal de K** :
  - Méthode du coude
  - Silhouette score
  - Gap statistic

### 2.2 DBSCAN
- **Paramètres cruciaux** :
  - eps (ε) : rayon du voisinage
  - min_samples : points minimum pour former un cluster
- **Types de points** :
  - Core : ont suffisamment de voisins
  - Border : proche d'un core point
  - Noise : ni core ni border
- **Avantages détaillés** :
  - Trouve des clusters de forme arbitraire
  - Résistant au bruit
  - Ne nécessite pas de K
- **Complexité** : $$O(n \log n)$$ avec indexation spatiale

### 2.3 PCA
- **Mathématiques sous-jacentes** :
  1. Centrage des données
  2. Calcul de la matrice de covariance
  3. Décomposition en valeurs propres
  4. Projection sur les premiers vecteurs propres
- **Choix du nombre de composantes** :
  - Variance expliquée cumulée
  - Scree plot
  - Kaiser criterion (valeurs propres > 1)
- **Extensions** :
  - Kernel PCA
  - Incremental PCA
  - Sparse PCA

## 3. **Apprentissage Profond**

### 3.1 Réseaux de Neurones Feed-Forward
- **Architecture détaillée** :
  - Couches d'entrée : dimension des features
  - Couches cachées : nombre variable
  - Couche de sortie : dimension de la prédiction
- **Fonctions d'activation** :
  - ReLU : $$f(x) = \max(0,x)$$
  - Sigmoid : $$f(x) = \frac{1}{1+e^{-x}}$$
  - Tanh : $$f(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$$
- **Techniques d'optimisation** :
  - SGD avec momentum
  - Adam
  - RMSprop
- **Régularisation** :
  - Dropout
  - Batch Normalization
  - L1/L2

### 3.2 CNN
- **Couches principales** :
  - Convolution : filtres de détection
  - Pooling : réduction dimensionnelle
  - Fully Connected : classification finale
- **Architectures populaires** :
  - ResNet : connexions résiduelles
  - VGG : blocs convolutifs profonds
  - Inception : convolutions parallèles
- **Techniques avancées** :
  - Transfer Learning
  - Data Augmentation
  - Feature Visualization

### 3.3 RNN et LSTM
- **Structure LSTM** :
  - Forget gate
  - Input gate
  - Output gate
  - Cell state
- **Variantes** :
  - GRU : version simplifiée
  - Bidirectional : contexte passé et futur
  - Attention mechanism
- **Applications spécifiques** :
  - Seq2Seq pour traduction
  - Encoder-Decoder
  - Time series forecasting

## **4. Métriques d'Évaluation Détaillées**

### 4.1 Classification

#### Exemple Concret : Évaluation d'un Modèle de Diagnostic Médical
```python
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Données exemple (vrais diagnostics vs prédictions)
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # 1: Malade, 0: Sain
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
y_proba = [0.8, 0.2, 0.6, 0.9, 0.3, 0.7, 0.8, 0.2, 0.9, 0.8]

# 1. Matrice de Confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sain', 'Malade'],
            yticklabels=['Sain', 'Malade'])
plt.title('Matrice de Confusion')
plt.ylabel('Réel')
plt.xlabel('Prédit')

# 2. Métriques détaillées
rapport = classification_report(y_true, y_pred)
print("\nRapport de Classification :")
print(rapport)

# 3. Courbe ROC
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")

# 4. Calculs détaillés
TP = cm[1,1]  # Vrais Positifs
TN = cm[0,0]  # Vrais Négatifs
FP = cm[0,1]  # Faux Positifs
FN = cm[1,0]  # Faux Négatifs

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nMétriques détaillées :")
print(f"Précision : {precision:.3f}")
print(f"Recall : {recall:.3f}")
print(f"F1-Score : {f1:.3f}")
print(f"Accuracy : {accuracy:.3f}")

# 5. Implications métier
cout_fp = 100  # Coût d'un faux positif (traitement inutile)
cout_fn = 500  # Coût d'un faux négatif (maladie non détectée)
cout_total = FP * cout_fp + FN * cout_fn

print(f"\nCoût total des erreurs : {cout_total}€")
```

#### Interprétation des Résultats :
1. **Matrice de Confusion** :
   - Diagonale principale (TP, TN) : bonnes prédictions
   - Hors diagonale (FP, FN) : erreurs

2. **Implications Pratiques** :
   - Précision élevée → Peu de faux positifs → Économie de traitements inutiles
   - Recall élevé → Peu de faux négatifs → Peu de malades non détectés
   - AUC-ROC proche de 1 → Bon pouvoir discriminant

3. **Choix du Seuil** :
   - Augmenter le seuil → Plus restrictif → Moins de faux positifs mais plus de faux négatifs
   - Diminuer le seuil → Plus permissif → Plus de détection mais plus de fausses alertes
- **Matrice de confusion** :
  ```
  |        | Pred + | Pred - |
  |--------|---------|---------|
  | True + |   TP    |   FN    |
  | True - |   FP    |   TN    |
  ```
- **Métriques dérivées** :
  - Précision = TP/(TP+FP)
  - Recall = TP/(TP+FN)
  - F1 = 2*(Précision*Recall)/(Précision+Recall)
  - ROC-AUC : aire sous la courbe ROC

### 4.2 Régression

#### Exemple Concret : Évaluation d'un Modèle de Prévision des Ventes
```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Données exemple (ventes réelles vs prédites)
ventes_reelles = [100, 150, 120, 180, 200, 160, 130, 210, 190, 140]
ventes_predites = [110, 155, 115, 170, 190, 165, 140, 200, 180, 150]

# 1. Calcul des métriques
mse = mean_squared_error(ventes_reelles, ventes_predites)
rmse = np.sqrt(mse)
mae = mean_absolute_error(ventes_reelles, ventes_predites)
r2 = r2_score(ventes_reelles, ventes_predites)

# 2. Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.scatter(ventes_reelles, ventes_predites, color='blue', alpha=0.5)
plt.plot([min(ventes_reelles), max(ventes_reelles)], 
         [min(ventes_reelles), max(ventes_reelles)], 
         'r--', lw=2)
plt.xlabel('Ventes Réelles')
plt.ylabel('Ventes Prédites')
plt.title('Prédictions vs Réalité')

# 3. Analyse des résidus
residus = np.array(ventes_reelles) - np.array(ventes_predites)
plt.figure(figsize=(10, 6))
plt.scatter(ventes_predites, residus, color='green', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Ventes Prédites')
plt.ylabel('Résidus')
plt.title('Analyse des Résidus')

# 4. Affichage des métriques
print("\nMétriques de performance :")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R² : {r2:.3f}")

# 5. Analyse des erreurs relatives
erreurs_relatives = np.abs(residus / np.array(ventes_reelles)) * 100
print(f"\nErreur relative moyenne : {np.mean(erreurs_relatives):.1f}%")
print(f"Erreur relative médiane : {np.median(erreurs_relatives):.1f}%")
print(f"Erreur relative max : {np.max(erreurs_relatives):.1f}%")

# 6. Implications business
seuil_alerte = 20  # Pourcentage d'erreur acceptable
erreurs_importantes = erreurs_relatives > seuil_alerte
print(f"\nNombre de prédictions avec erreur > {seuil_alerte}% : "
      f"{sum(erreurs_importantes)}")

# 7. Distribution des erreurs
plt.figure(figsize=(10, 6))
plt.hist(erreurs_relatives, bins=20, color='purple', alpha=0.6)
plt.axvline(np.mean(erreurs_relatives), color='r', linestyle='--',
            label=f'Moyenne ({np.mean(erreurs_relatives):.1f}%)')
plt.xlabel('Erreur Relative (%)')
plt.ylabel('Fréquence')
plt.title('Distribution des Erreurs Relatives')
plt.legend()
```

#### Interprétation des Résultats :

1. **RMSE vs MAE** :
   - RMSE > MAE → Présence d'erreurs importantes
   - RMSE ≈ MAE → Erreurs uniformes

2. **R² et Business** :
   - R² = 0.95 → 95% de la variance expliquée
   - Utile pour la confiance dans les prédictions

3. **Analyse des Résidus** :
   - Pattern visible → Biais systématique
   - Distribution aléatoire → Bon modèle

4. **Impact Business** :
   - Coût des sous-estimations (rupture de stock)
   - Coût des surestimations (surstock)
   - Ajustement des seuils d'alerte
- **Formules détaillées** :
  - MSE = $$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
  - RMSE = $$\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
  - MAE = $$\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
  - R² = $$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$


## **5. Outils et Frameworks**

### 5.1 Scikit-learn
- **Avantages** :
  - API cohérente
  - Documentation excellente
  - Intégration facile
- **Modules principaux** :
  - preprocessing
  - model_selection
  - metrics
  - pipeline

### 5.2 TensorFlow/Keras
- **Architecture** :
  - Graphe de calcul
  - Eager execution
  - Distribution possible
- **Fonctionnalités** :
  - Custom training loops
  - TF.data pour les pipelines
  - TensorBoard pour visualisation

### 5.3 PyTorch
- **Caractéristiques** :
  - Dynamic computational graphs
  - Pythonic
  - Debugging facile
- **Écosystème** :
  - torchvision
  - torchaudio
  - torchtext
