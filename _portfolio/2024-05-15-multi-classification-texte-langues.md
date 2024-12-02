---
title: "Multi-classification de Texte (Langues)"
date: 2024-05-15
collection: portfolio
permalink: /portfolio/2024-05-15-multi-classification-texte-langues
categories: [Projet Étudiant, NLP, Classification, Analyse Multilingue]
tags: [Python, Selenium, Stanza, SpaCy, XLM-RoBERTa, SMOTE]
description: "Classification automatique de textes multilingues en fonction de leur langue d'origine, avec extraction de données de Wikipédia, analyse morphosyntaxique et entraînement d'un modèle de classification multilingue."
---

### Contexte du Projet

Ce projet visait à classifier automatiquement des textes multilingues en fonction de leur langue d'origine, en utilisant un corpus extrait de pages Wikipédia pour plusieurs langues. Une analyse approfondie a été menée sur la structure linguistique et les distributions lexicales de chaque langue, suivie d'une évaluation de la performance d'un modèle de classification multilingue.

---

### Missions Réalisées

1. **Extraction de Données Textuelles Multilingues :**
   - Utilisation de **Selenium** pour extraire automatiquement des articles de Wikipédia dans plusieurs langues, en collectant des paragraphes aléatoires et en explorant les liens interlangues pour enrichir le corpus.
   - Création d'une base de données multilingue, avec un nombre équilibré d'articles pour chaque langue cible, enregistrés dans des fichiers texte.

2. **Nettoyage et Prétraitement des Données :**
   - Suppression des éléments non désirés (numéros de lignes, chiffres isolés, retours à la ligne).
   - Utilisation de modèles NLP multilingues (Stanza et SpaCy) pour segmenter les textes en phrases, puis regroupement en paires de phrases par langue dans un fichier CSV équilibré.
   - Nettoyage des valeurs nulles et gestion des valeurs aberrantes basées sur la longueur des textes pour assurer une qualité de données optimale.

3. **Analyse Morphosyntaxique et Lexicale :**
   - Calcul de la distribution des mots et de la diversité lexicale par langue, avec l’utilisation de la **loi de Zipf** pour explorer la fréquence des mots.
   - Calcul de la **corrélation entre la longueur des textes et la diversité lexicale** pour chaque langue afin d’identifier des différences structurelles.
   - Analyse des entités nommées et des catégories grammaticales par langue, stockées pour de futures analyses.

4. **Entraînement et Évaluation du Modèle de Classification Multilingue :**
   - Entraînement d'un modèle basé sur **XLM-RoBERTa** pour la classification des langues, avec des métriques d’évaluation standard (précision, rappel, F1-score) et matrice de confusion pour évaluer les performances.
   - Augmentation des étiquettes sous-représentées avec **SMOTE** pour équilibrer les données et améliorer la performance du modèle sur les langues moins fréquentes.

---

### Compétences Techniques Acquises

- **Selenium** pour l'extraction automatisée de données multilingues depuis Wikipédia.
- **Pandas** pour la structuration et le nettoyage des données.
- **Stanza et SpaCy** pour l'analyse morphosyntaxique et la lemmatisation des textes multilingues.
- **NumPy et Matplotlib** pour le traitement des données et la visualisation des résultats.
- **XLM-RoBERTa** pour la classification multilingue et **Scikit-learn** pour l’évaluation des performances.
- **SMOTE** pour l'augmentation des données, en équilibrant les classes minoritaires.

---

### Compétences Humaines Acquises

- **Autonomie :** Gestion indépendante de la collecte, du nettoyage et de l'analyse des données.
- **Capacité de Recherche :** Exploration de sources multilingues et adaptation des méthodes de traitement pour chaque langue.
- **Esprit d'Analyse et de Synthèse :** Interprétation des résultats et formulation de conclusions basées sur les performances du modèle et les distributions linguistiques observées.

---

### Résultats et Éléments de Preuve

- **Modèle de Classification Multilingue :** Le modèle basé sur XLM-RoBERTa a atteint des scores élevés de précision et de F1, démontrant une capacité robuste à distinguer les langues.
- **Visualisations des Distributions Lexicales et Loi de Zipf :**
   - Graphiques illustrant la loi de Zipf pour chaque langue, montrant les différences de fréquence d’utilisation des mots.
   - Corrélations entre la longueur des textes et la diversité lexicale, permettant d’identifier la richesse linguistique de chaque langue dans le corpus.
- **Exemples de Prédictions et Matrice de Confusion :** Résultats des prédictions du modèle et des erreurs fréquentes visualisées dans une matrice de confusion, montrant les langues parfois confondues et les possibilités d'amélioration.

> **[Télécharger le rapport final](https://github.com/PerrineQhn/OutilsTraitementCorpus/blob/main/README.md)** - Rapport détaillant le processus, les résultats et les interprétations des données analysées.
>
<!-- > **[Voir les visualisations](#)** - Accédez aux graphiques de la loi de Zipf, de la diversité lexicale et des performances de classification. -->
>
> **[Lien vers le dépôt GitHub](https://github.com/PerrineQhn/OutilsTraitementCorpus)** - Dépôt GitHub pour reproduire l’extraction, le prétraitement et la classification des données multilingues.

---

Ce projet m'a permis de développer une approche complète et rigoureuse pour la classification de textes multilingues, en renforçant mes compétences en extraction et traitement de données, ainsi qu'en analyse de modèles de classification. Ce travail m'a aussi sensibilisé aux spécificités linguistiques des langues et aux défis des données multilingues en NLP.

--- 