---
title: "Création d'un OCR Chinois"
date: 2024-10-10
collection: portfolio
permalink: /portfolio/2024-10-10-ocr-chinois
categories: [Projet Étudiant, OCR, Deep Learning, Travail en Équipe]
tags: [Python, PyTorch, Tesseract, EasyOCR, Traitement d'Images, LSTM, CNN]
description: "Développement d'un OCR pour les caractères chinois, avec une attention particulière aux caractères manuscrits, en utilisant le modèle CRNN et des techniques de prétraitement avancées."

---

# Création d'un OCR Chinois (octobre 2024)

### Contexte du Projet

Ce projet étudiant visait à développer un OCR (reconnaissance optique de caractères) pour les caractères chinois, avec une attention particulière aux caractères manuscrits. Le modèle CRNN, associé à des techniques de prétraitement avancées, a permis d'améliorer la précision de la reconnaissance pour ce type d'écriture.

---

### Missions Réalisées

1. **Prétraitement des Images d’Entraînement :**
   - **Détection et Correction d'Inclinaison :** Un algorithme de redressement détecte l'angle de déformation de chaque image pour aligner le texte horizontalement. Cette étape améliore la précision en éliminant les variations d'inclinaison.
   - **Segmentation Ligne par Ligne :** Les images sont segmentées ligne par ligne en extrayant uniquement les formes significatives, facilitant la reconnaissance des caractères.
   - **Réduction du Bruit :** Utilisation d'opérations morphologiques et de flou médian pour éliminer le bruit et obtenir des images plus nettes, prêtes pour l'analyse.

2. **Création et Préparation d’un Corpus de Caractères Chinois :**
   - Constitution d'un corpus de référence riche et varié en caractères chinois, issu de bases de données publiques, pour fournir un large éventail d'exemples au modèle.

3. **Développement d’un Modèle CRNN avec PyTorch :**
   - Le modèle CRNN (Convolutional Recurrent Neural Network) combine un CNN pour l'extraction des caractéristiques visuelles avec un LSTM pour capturer la séquence des caractères. Cette architecture est particulièrement efficace pour reconnaître les caractères manuscrits.

4. **Comparaison avec Tesseract et EasyOCR :**
   - Les performances du modèle CRNN ont été comparées avec les solutions **Tesseract** et **EasyOCR** pour évaluer les gains de précision apportés par notre approche.

5. **Évaluation des Performances :**
   - Calcul de métriques de classification comme la précision, le rappel et le F1-score, et génération de matrices de confusion pour identifier les erreurs fréquentes et guider les améliorations.

6. **Test sur un Nouveau Dataset :**
   - Le modèle a été testé sur un autre jeu de données (Guilhem) pour évaluer sa robustesse et ses capacités de généralisation.

---

### Compétences Techniques Acquises

- **Python** pour le développement des scripts d’entraînement et de prétraitement.
- **PyTorch** pour le développement et l'entraînement du modèle CRNN.
- **Tesseract** et **EasyOCR** pour la comparaison des performances avec des solutions existantes.
- **Scikit-learn** pour l’analyse des performances avec des métriques comme la précision et le F1-score.
- **Matplotlib** et **Pandas** pour visualiser les matrices de confusion et analyser les résultats.

---

### Compétences Humaines Acquises

- **Travail en Équipe :** Collaboration pour le développement et l'évaluation des résultats.
- **Autonomie :** Préparation et ajustement des paramètres pour obtenir des résultats optimaux en OCR.
- **Capacité d'Analyse :** Analyse approfondie des résultats pour identifier les forces et faiblesses de chaque méthode.

---

### Résultats et Éléments de Preuve

- **Amélioration des Performances :** Le modèle CRNN a montré une amélioration significative de la précision et du F1-score comparé aux outils Tesseract et EasyOCR, en particulier sur les caractères manuscrits chinois.
- **Tableau Comparatif des Métriques :**
   - | Méthode       | Précision | Rappel | F1-Score |
     |---------------|-----------|--------|----------|
     | Tesseract     | 0.78      | 0.75   | 0.76     |
     | EasyOCR       | 0.83      | 0.80   | 0.81     |
     | **CRNN**      | **0.91**  | **0.88** | **0.89** |

- **Visualisation des Erreurs :** Création de matrices de confusion pour identifier les erreurs fréquentes, offrant une base pour des améliorations futures.

<!-- - **Illustrations :**
   - Avant et après prétraitement (réduction du bruit, redressement) et prédictions du modèle CRNN pour divers caractères.

> **[Télécharger le rapport final](#)** - Rapport complet détaillant la méthodologie, les résultats et les recommandations.
>
> **[Voir la démonstration](#)** - Accédez à la démonstration de l'OCR en ligne pour tester les capacités de reconnaissance en direct.
>
> **[Lien vers le code GitHub](#)** - Code source et instructions pour reproduire les résultats. -->

---

Ce projet a permis d'approfondir mes connaissances en reconnaissance de caractères, en traitement d'image et en évaluation de modèles de deep learning, tout en travaillant de manière collaborative et autonome pour résoudre des défis techniques complexes.

--- 