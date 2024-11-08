---
title: "Synthétiseur Vocal"
date: 2024-01-15
collection: portfolio
client: "Projet Étudiant"
permalink: /portfolio/2024-01-15-synthetiseur-vocal
categories: [Projet Étudiant, Traitement du Signal, NLP]
tags: [Python, Praat, Parselmouth, Synthèse Vocale, Diphones]
description: "Création d'un synthétiseur vocal à partir d'un ensemble de diphones enregistrés et annotés, pour transformer du texte en voix synthétique."
---

### Contexte du Projet

Ce projet étudiant avait pour objectif de créer un synthétiseur vocal en utilisant un ensemble de diphones enregistrés et annotés. Grâce aux outils de traitement de signal et à la synthèse vocale, le projet permet de transformer du texte en une voix synthétique avec une prononciation précise et contrôlée.

---

### Missions Réalisées

1. **Enregistrement et Annotation des Diphones :**
   - Utilisation de **Praat** et de fichiers **TextGrid** pour enregistrer les diphones et les annoter en respectant les conventions de durée et de qualité sonore. Chaque diphone est segmenté pour une utilisation précise lors de la synthèse.

2. **Intégration d'un Dictionnaire SAMPA pour la Prononciation :**
   - Utilisation d’un dictionnaire de prononciation en alphabet phonétique SAMPA pour convertir les mots français en transcriptions phonétiques. Ce dictionnaire guide l’assemblage des phonèmes pour une synthèse vocale cohérente.

3. **Synthèse Vocale avec Python :**
   - Création d’un script en **Python** utilisant **Parselmouth** pour lire les annotations et concaténer les diphones. Les segments sont ajustés en durée pour refléter la prosodie naturelle.

4. **Gestion des Phonèmes et Durées :**
   - Contrôle de la durée des voyelles longues et des consonnes pour une fluidité dans la synthèse. Les fenêtres rectangulaires sont utilisées pour concaténer les sons, en ajustant chaque phonème pour un rendu vocal naturel.

5. **Préparation des Fichiers Textes et Configuration du Script :**
   - Les fichiers `.wav` et `.TextGrid` sont chargés pour les diphones, tandis que les phrases sont introduites via un fichier texte pour être synthétisées en voix.

---

### Compétences Techniques Acquises

- **Praat et TextGrid** : Outils pour l'enregistrement, l'annotation et la segmentation des diphones.
- **Parselmouth** : Interface Python pour Praat, utilisée pour manipuler les diphones et créer une voix synthétique.
- **Python** : Développement du script de synthèse vocale et gestion de l’entrée/sortie des fichiers, avec intégration du dictionnaire SAMPA.

---

### Compétences Humaines Acquises

- **Autonomie** : Gestion du processus d'enregistrement, d'annotation et de synthèse de manière indépendante.
- **Rigueur** : Attention aux détails dans la manipulation des diphones et l’annotation, assurant la cohérence de la synthèse vocale.
- **Capacité de Résolution de Problèmes** : Ajustement des phonèmes et de la durée des segments pour optimiser la clarté et le naturel de la voix synthétique.

---

### Résultats et Éléments de Preuve

- **Synthèse Vocale Fonctionnelle :** Le script Python produit une voix synthétique cohérente avec les diphones enregistrés, démontrant une capacité à lire des phrases en respectant les règles de prosodie.
- **Exemples Audio :** Des fichiers audio générés illustrent la précision de la synthèse pour plusieurs phrases testées.

> **[Télécharger le script Python](#)** - Accédez au code source utilisé pour la synthèse vocale.
>
> **[Écouter un exemple audio](#)** - Écoutez un extrait de la voix synthétique générée par le projet.
>
> **[Lien vers le dépôt GitHub](#)** - Code complet et instructions pour reproduire le projet.

---

Ce projet a été une excellente introduction au traitement de signal et à la synthèse vocale, me permettant de renforcer mes compétences en annotation linguistique et en programmation pour le traitement du signal audio. L'utilisation de Praat et Parselmouth a particulièrement enrichi mes connaissances en manipulation de signaux vocaux et en génération de contenu vocal.

--- 