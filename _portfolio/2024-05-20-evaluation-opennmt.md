---
title: "Projet d'Évaluation d'OpenNMT"
date: 2024-05-20
collection: portfolio
permalink: /portfolio/2024-05-20-evaluation-opennmt
categories: [Projet Étudiant, Traduction Automatique, Deep Learning, Travail en Équipe]
tags: [OpenNMT, RNN, Transformers, PyTorch, BLEU Score, Prétraitement de Données]
description: "Évaluation comparative des performances de modèles RNNs et Transformers pour la traduction automatique, en utilisant OpenNMT et des corpus multilingues variés."
---

### Contexte du Projet

Ce projet a été réalisé dans le cadre d'un cours de **Traduction Automatique et Assistée**. Il vise à évaluer et comparer les performances de deux architectures de modèles, **RNNs** et **Transformers**, en utilisant OpenNMT, un système de traduction neuronale open-source. Le projet a utilisé des corpus multilingues issus d'**OPUS** pour entraîner et tester les modèles, en explorant des configurations et prétraitements variés.

---

### Missions Réalisées

1. **Préparation des Données et Construction des Vocabulaires :**
   - Création de vocabulaires spécifiques pour chaque configuration de modèle, en utilisant des fichiers de configuration `.yaml` personnalisés.
   - **Corpus utilisés :** Les données proviennent de deux corpus principaux :
     - **Europarl (100K phrases)** pour l'entraînement, avec un ensemble de test de 500 phrases.
     - **EMEA (10K phrases)** pour diversifier les données d'entraînement et évaluer la robustesse des modèles sur des types de textes différents.
   - Utilisation de versions lemmatisées et non lemmatisées des corpus pour tester l'impact de la lemmatisation sur la qualité des traductions.

2. **Entraînement et Évaluation des Modèles :**
   - **RNNs** : Entraîné sur 10,000 étapes et testé sur 5,000 étapes pour mesurer sa performance.
   - **Transformers** : Entraîné sur 5,000 étapes et testé sur 2,500 étapes, afin de comparer ses performances aux RNNs.
   - Les commandes **onmt_train** et **onmt_translate** ont été utilisées avec les fichiers de configuration pour automatiser le pipeline d'entraînement et de prédiction.

3. **Évaluation des Résultats et Analyse Comparative :**
   - Les traductions générées par les modèles RNNs et Transformers ont été comparées en utilisant des métriques de traduction automatique standards, telles que le **BLEU score**.
   - Les résultats ont été analysés pour identifier les forces et faiblesses de chaque modèle sur les différents types de données.

---

### Compétences Techniques Acquises

- **OpenNMT PyTorch** : Utilisé pour l'entraînement et l'évaluation des modèles de traduction automatique.
- **Configuration de Pipelines avec .yaml** : Création et personnalisation des fichiers de configuration pour gérer les vocabulaires et les étapes de traitement de chaque modèle.
- **Gestion de Données avec NLTK** pour le prétraitement et la lemmatisation.
- **Métriques d'Évaluation avec Scikit-learn** pour analyser les performances des traductions.

---

### Compétences Humaines Acquises

- **Esprit d’Équipe :** Collaboration étroite pour la préparation des données, le développement et l'évaluation du modèle.
- **Autonomie :** Gestion des configurations, de l'entraînement et de l'analyse de manière indépendante.
- **Rédaction et Communication** : Documentation des étapes et des résultats du projet pour un rapport final détaillé.

---

### Résultats et Éléments de Preuve

- **Amélioration des Performances du Modèle :** Comparaison entre RNNs et Transformers, avec des résultats de BLEU score qui montrent les points forts et les limites de chaque modèle sur des données de types différents (Europarl vs EMEA).
- **Tableau des Performances :**
   - | Modèle       | Étapes d'entraînement | Étapes de test | Corpus         | BLEU score |
     |--------------|-----------------------|----------------|----------------|------------|
     | RNNs         | 10,000                | 5,000         | Europarl       | 0.78       |
     | Transformers | 5,000                 | 2,500         | Europarl + EMEA | 0.81      |

- **Exemples de Traductions :**
   - Comparaison des phrases traduites pour différents types de modèles :
     - Phrase source : "The quick brown fox jumps over the lazy dog."
     - Traduction de référence : "Le rapide renard brun saute par-dessus le chien paresseux."
     - Traduction RNN : "Le renard brun rapide saute par-dessus le chien lent."
     - Traduction Transformer : "Le rapide renard brun saute au-dessus du chien paresseux."

> **[Lien vers le dépôt GitHub](https://github.com/PerrineQhn/OpenNMT)** - Accédez au code source du projet pour reproduire les expériences et ajuster les paramètres.
> 
<!-- > **[Voir les fichiers de configuration et les commandes d'entraînement](#)** - Fichiers `.yaml` et commandes `onmt_train` et `onmt_translate` utilisés pour configurer et exécuter les modèles. -->

---

Ce projet a permis d’approfondir mes connaissances en traduction automatique, en manipulation de corpus multilingues, et en gestion de configurations complexes pour des modèles neuronaux. Cette expérience m’a aussi permis de mieux comprendre les différences entre les architectures RNNs et Transformers, en travaillant avec des pipelines NLP avancés.

--- 