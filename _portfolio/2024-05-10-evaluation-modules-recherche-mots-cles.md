---
title: "Évaluation de Modules pour la Recherche de Mots-Clés"
date: 2024-05-10
collection: portfolio
client: "Projet Étudiant"
permalink: /portfolio/2024-05-10-evaluation-modules-recherche-mots-cles
categories: [Projet Étudiant, NLP, Recherche de Mots-Clés, Travail en Groupe]
tags: [Python, Gensim, Scikit-learn, NLTK, Analyse de Corpus, Évaluation de Modèles]
description: "Évaluation comparative de méthodes de recherche de mots-clés (TF-IDF et LDA) sur des corpus de StackOverflow et Wikipedia, avec analyse des performances et des contextes d'application."
---

### Contexte du Projet

Ce projet avait pour objectif d'explorer et de comparer des méthodes de recherche de mots-clés dans des corpus textuels, en utilisant les modèles **TF-IDF** et **LDA** (Latent Dirichlet Allocation). Nous avons évalué les performances de chaque méthode sur deux ensembles de données : **StackOverflow** et **Wikipedia**. Cette étude comparative visait à déterminer dans quels contextes chaque modèle excelle pour identifier les termes les plus représentatifs.

---

### Missions Réalisées

1. **Préparation des Données et Prétraitement :**
   - Les corpus **StackOverflow** (20,000 lignes) et **Wikipedia** (13,014 lignes) ont été divisés en ensembles d’entraînement, de test, et de développement.
   - Prétraitement des données incluant la lemmatisation, la suppression de balises HTML, de caractères spéciaux et de mots vides. Pour **TF-IDF**, les mots présents dans moins de 20 documents ont été filtrés, tandis que pour **LDA**, la fréquence minimale était gérée via le paramètre `no_below`.
   - Extraction de **n-grammes** (bigrammes et trigrammes) pour enrichir la liste de mots-clés et capturer des termes fréquemment associés.

2. **Implémentation des Modèles et Extraction de Mots-Clés :**
   - **TF-IDF** : Calcul des scores de termes basés sur la fréquence inverse des documents pour identifier les mots-clés spécifiques à chaque document.
   - **LDA** : Modélisation des documents comme des combinaisons de topics, permettant de capturer des thèmes généraux dans le corpus. Le modèle a été configuré pour effectuer 200 itérations d'assignation de topics, avec les mots les plus probables de chaque topic récupérés comme mots-clés.

3. **Évaluation des Résultats :**
   - Mesures de précision, rappel et F-mesure, calculées pour chaque méthode sur les deux corpus, afin de comparer leur capacité à extraire des mots-clés pertinents.
   - Analyse des ratios total match/total tags de référence, mesurant la proportion de mots-clés extraits par chaque modèle correspondant aux mots-clés de référence dans le corpus.

---

### Compétences Techniques Acquises

- **Gensim** pour l’implémentation de **LDA** et la modélisation de topics.
- **Scikit-learn** pour les calculs **TF-IDF** et les évaluations de performance.
- **NLTK** pour le nettoyage et la préparation des textes.
- **Pandas** pour la gestion et l’analyse des données tabulaires.
- **Calcul de n-grammes** pour enrichir l'analyse des cooccurrences.

---

### Compétences Humaines Acquises

- **Esprit d’Équipe :** Collaboration étroite pour définir les critères d’évaluation, harmoniser le prétraitement des données et interpréter les résultats.
- **Autonomie :** Recherche indépendante sur les meilleures pratiques pour l'extraction de mots-clés avec des modèles non supervisés.
- **Capacité d’Analyse :** Évaluation critique des résultats et identification des contextes dans lesquels chaque modèle offre les meilleures performances.

---

### Résultats et Éléments de Preuve

- **Comparaison des Méthodes :** 
   - Sur le corpus **StackOverflow**, **TF-IDF** a montré de meilleures performances globales :
      - Précision : TF-IDF 3.1% vs LDA 1.48%
      - Rappel : TF-IDF 20.4% vs LDA 16.16%
      - F-mesure : TF-IDF 5.16% vs LDA 2.57%
   - Sur le corpus **Wikipedia**, les performances des deux modèles étaient plus comparables, avec **LDA** légèrement supérieur :
      - Précision : LDA 6.11% vs TF-IDF 5.69%
      - Rappel : LDA 14.72% vs TF-IDF 13.07%
      - F-mesure : LDA 7.82% vs TF-IDF 7.05%

- **Analyse des Ratios de Correspondance** :
   - StackOverflow : LDA 14.64% vs TF-IDF 19.76%
   - Wikipedia : LDA 13.65% vs TF-IDF 11.48%

> **[Télécharger le rapport final](#)** - Rapport détaillé documentant la méthodologie, les résultats et les interprétations.
>
<!-- > **[Accéder aux notebooks](#)** - Visualisez les notebooks pour l'implémentation de LDA et TF-IDF, ainsi que les résultats de chaque méthode. -->
>
> **[Lien vers le dépôt GitHub](#)** - Code source pour reproduire l’extraction et l’évaluation des mots-clés.

---

Ce projet a renforcé mes compétences en extraction de mots-clés et en évaluation de modèles, tout en offrant une perspective comparative entre des techniques classiques de recherche de thèmes et de mots-clés. Les résultats ont permis de comprendre les forces et limites de chaque approche, ce qui peut être appliqué dans des projets de NLP plus avancés.

---