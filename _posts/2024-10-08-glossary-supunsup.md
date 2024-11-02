---
title: 'Supervisé vs non supervisé'
date: 2014-10-08
#modified: 
permalink: /machine-learning-glossary/concepts/supunsup
toc: false
excerpt: " Concepts en apprentissage automatique : apprentissage supervisé vs apprentissage non supervisé."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
author_profile: false
redirect_from: 
  - /posts/2014/10/glossary-supunsup
sidebar:
  title: "Glossaire ML"
  nav: sidebar-glossary
---

{% include base_path %}



## **Apprentissage supervisé**

*Les tâches d'apprentissage supervisé s'attaquent aux problèmes où nous disposons de données étiquetées.*

:bulb: <span class='intuition'>**Intuition :** On peut le voir comme un enseignant qui corrige un examen à choix multiples. </span> À la fin, vous recevrez votre note moyenne ainsi que les réponses correctes à toutes les questions. 

L’apprentissage supervisé peut être divisé en deux grands types de problèmes :
* **Classification**: ici, la variable de sortie $y$ est catégorielle. Nous essayons essentiellement d’attribuer une ou plusieurs classes à une observation. Exemple : est-ce un chat ou pas ?
* **Regression**: ici, la variable de sortie $y$ est continue. Exemple : quelle est la taille de cette personne ?

### Classification
*Le problème de classification consiste à attribuer un ensemble de classes/catégories à une observation. C’est-à-dire* $$\mathbf{x} \mapsto y,\ y \in \{0,1,...,C\}$$*

Les problèmes de classification peuvent être subdivisés en :

* **Binaire:** Il y a 2 classes possibles. $$C=2,\ y \in \{0,1\}$$
* **Multi-classes :** Il y a plus de 2 classes possibles. $$C>2$$
* **Multi-étiquettes :** Si les étiquettes ne sont pas mutuellement exclusives. Cela est souvent remplacé par $$C$$ classifications binaires spécifiant si une observation doit être assignée à chaque classe.

Les mesures d'évaluation courantes incluent la précision, le F1-Score, l'AUC... J'ai une [section dédiée à ces métriques de classification](#classification-metrics).

## **Apprentissage non supervisé**

*Les tâches d’apprentissage non supervisé consistent à trouver des structures dans des données **non étiquetées** sans résultat désiré spécifique.*

L’apprentissage non supervisé peut être divisé en plusieurs sous-tâches (la séparation n’est pas aussi claire que dans le cadre supervisé) :
* **Clustering :** pouvez-vous trouver des clusters dans les données ?
* **Estimation de densité :** quelle est la distribution de probabilité sous-jacente qui a donné lieu aux données ?
* **Réduction de dimensionnalité :** comment mieux compresser les données ?
* **Détection d’anomalies :** quels points de données sont des anomalies ?

En raison du manque de labels de vérité terrain, il est difficile de mesurer les performances de ces méthodes, mais elles sont extrêmement importantes en raison de la quantité de données non étiquetées accessibles.
