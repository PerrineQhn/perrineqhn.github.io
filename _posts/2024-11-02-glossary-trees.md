---
title: 'Decision trees'
date: 2024-11-02
modified: 2024-11-02
permalink: /machine-learning-glossary/models/trees
toc: false
excerpt: "ML concepts: decision trees."
header: 
  teaser: "blog/glossary/glossary.png"
tags:
  - ML
  - Glossary
redirect_from: 
  - /posts/2017/11/glossary-trees/
author_profile: false
sidebar:
  title: "ML Glossary"
  nav: sidebar-glossary
---

{% include base_path %}


<div>
<details open>

<div markdown='1'>
* :bulb: <span class='intuition'> Intuition </span> :
    * Divisez les données d'entraînement en fonction de "la meilleure" question (*e.g.* est-il plus âgé que 27 ans ?). Divisez récursivement les données tant que vous n'êtes pas satisfait des résultats de classification.
    * Les arbres de décision sont essentiellement l'algorithme à utiliser pour le jeu des "20 questions". [Akinator](http://en.akinator.com/) est un bon exemple de ce qui peut être implémenté avec des arbres de décision. Akinator est probablement basé sur des systèmes experts de logique floue (car il peut fonctionner avec des réponses erronées), mais vous pourriez faire une version plus simple avec des arbres de décision.
    * Les divisions "optimales" sont trouvées par maximisation du [gain d'information](/machine-learning-glossary/information/#machine-learning-and-entropy) ou des méthodes similaires.
* :wrench: <span class='practice'> Pratique </span> :
    * Les arbres de décision sont performants lorsque vous avez besoin d'un modèle simple et interprétable, mais que la relation entre $y$ et $\mathbf{x}$ est complexe.
    * Complexité d'entraînement : <span class='practiceText' markdown='1'> $O(MND + ND\log(N) )$ </span>.
    * Complexité de test : <span class='practiceText' markdown='1'> $O(MT)$ </span>.
    * Notation utilisée : $M=profondeur$ ; $$N= \#_{train}$$ ; $$D= \#_{features}$$ ; $$T= \#_{test}$$.
* :white_check_mark: <span class='advantage'> Avantage </span> :
    * <span class='advantageText'>  Interprétable </span>.
    * Peu d'hyper-paramètres.
    * Nécessite moins de nettoyage de données :
        * Pas de normalisation nécessaire.
        * Peut gérer les valeurs manquantes.
        * Gère les variables numériques et catégorielles.
    * Robuste aux valeurs aberrantes.
    * Ne fait pas d'hypothèses sur la distribution des données.
    * Effectue une sélection de caractéristiques.
    * S'adapte bien à l'échelle.
* :x: <span class='disadvantage'> Inconvénient </span> :
    * Précision généralement faible en raison de la sélection gloutonne.
    * <span class='disadvantageText'> Forte variance </span> car si la première division change, tout le reste change.
    * Les divisions sont parallèles aux axes des caractéristiques => besoin de multiples divisions pour séparer 2 classes avec une frontière de décision à 45°.
    * Pas d'apprentissage en ligne.
</div>
</details>
</div> 
<p></p>

## Classification 

Les arbres de décision sont plus souvent utilisés pour les problèmes de classification, donc nous allons nous concentrer sur ce contexte pour l'instant.

L'idée de base pour construire un arbre de décision est de :
1. Trouver une division optimale (caractéristique + seuil). *C'est-à-dire* la division qui minimise l'impureté (maximise le gain d'information).
2. Partitionner le jeu de données en 2 sous-ensembles en fonction de la division ci-dessus.
3. Appliquer récursivement $1$ et $2$ à chaque nouveau sous-ensemble jusqu'à ce qu'un critère d'arrêt soit atteint.
4. Pour éviter le sur-apprentissage : élaguer les nœuds qui "ne sont pas très utiles".

Voici un petit gif montrant ces étapes :
<div markdown="1">
![Construction des arbres de décision pour la classification](/images/blog/glossary-old/decision-tree-class.gif){:width="477px" height="358px"}
</div>

Note : Pour plus d'informations, veuillez consulter les sections "*détails*" et "*Pseudocode et Complexité*" ci-dessous.

<div>
<details>
<summary>Détails</summary> 
<div markdown='1'>
L'idée derrière les arbres de décision est de partitionner l'espace d'entrée en plusieurs régions. *Par exemple* : région des hommes de plus de 27 ans. Ensuite, prédire la classe la plus probable pour chaque région en assignant la modalité des données d'entraînement dans cette région. Malheureusement, trouver une partition optimale est généralement infaisable en termes de calcul ([NP-complet](https://people.csail.mit.edu/rivest/HyafilRivest-ConstructingOptimalBinaryDecisionTreesIsNPComplete.pdf)) en raison du nombre combinatoire de possibles arbres. En pratique, les différents algorithmes utilisent donc une approche gloutonne. *C'est-à-dire* chaque division de l'arbre de décision essaie de maximiser un certain critère sans tenir compte des divisions suivantes.

*Comment définir un critère d'optimalité pour une division ?* Définissons une impureté (erreur) de l'état actuel, que nous allons essayer de minimiser. Voici 3 impuretés d'état possibles :

* **Erreur de Classification** :  
    * :bulb: <span class='intuitionText'> L'erreur de précision : $1-Acc$</span> de l'état actuel. *C'est-à-dire* l'erreur que nous commettrions en nous arrêtant à l'état actuel.
    * $$ErreurClassification = 1 - \max_c (p(c))$$

* **[Entropie](/machine-learning-glossary/information/#entropy)** :  
    * :bulb: <span class='intuitionText'> À quel point les classes de l'état actuel sont-elles imprévisibles</span>.
    * Minimiser l'entropie correspond à maximiser le [gain d'information](/machine-learning-glossary/information/#machine-learning-and-entropy).
    * $$Entropie = - \sum_{c=1}^C p(c) \log_2 \ p(c)$$

* **Impureté de Gini** :  
    * :bulb: <span class='intuitionText'> Probabilité attendue ($\mathbb{E}[\cdot] = \sum_{c=1}^C p(c) (\cdot) $) de mauvaise classification ($\sum_{c=1}^C p(c) (1-\cdot)$) d'un élément sélectionné au hasard, s'il était classé selon la distribution des étiquettes ($\sum_{c=1}^C p(c) (1-p(c))$)</span>.
    * $$ErreurClassification =  \sum_c^C p_c (1-p_c) = 1- \sum_c^C p_c^2$$

Voici un graphique rapide montrant l'impureté en fonction de la distribution des classes dans un cadre binaire :

<div markdown='1'>
![Mesure d'Impureté](/images/blog/glossary-old/impurity.png){:width='477px'}
</div>

:mag: <span class='note'> Notes annexes </span> :

* L'erreur de classification peut sembler un choix naturel, mais ne vous laissez pas tromper par les apparences : elle est généralement moins performante que les 2 autres méthodes :
    * Elle est "plus" gloutonne que les autres. En effet, elle ne se concentre que sur l'erreur actuelle, tandis que Gini et l'Entropie essaient de faire une division plus pure qui facilitera les étapes suivantes. <span class='exampleText'> Supposons que nous ayons une classification binaire avec 100 observations dans chaque classe $(100,100)$. Comparons une division qui sépare les données en $(20,80)$ et $(80,20)$, à une autre qui les divise en $(40,100)$ et $(60,0)$. Dans les deux cas, l'erreur de précision serait de $0,20\%$. Mais nous préférerions le deuxième cas, qui est **pur** et n'aura pas besoin d'être divisé davantage. L'impureté de Gini et l'Entropie choisiraient correctement ce dernier. </span> 
    * L'erreur de classification ne prend en compte que la classe la plus probable. Ainsi, avoir une division avec 2 classes extrêmement probables aura une erreur similaire à une division avec une classe extrêmement probable et plusieurs classes improbables.
* L'impureté de Gini et l'Entropie [diffèrent moins de 2% du temps](https://www.unine.ch/files/live/sites/imi/files/shared/documents/papers/Gini_index_fulltext.pdf) comme vous pouvez le voir dans le graphique ci-dessus. L'Entropie est un peu plus lente à calculer en raison de l'opération logarithmique.

**Quand devons-nous arrêter de diviser ?** Il est important de ne pas diviser trop de fois pour éviter le sur-apprentissage. Voici quelques heuristiques qui peuvent être utilisées comme critère d'arrêt :

* Lorsque le nombre d'exemples d'entraînement dans un nœud feuille est faible.
* Lorsque la profondeur atteint un seuil.
* Lorsque l'impureté est faible.
* Lorsque le gain de pureté dû à la division est faible.

Ces heuristiques nécessitent des seuils dépendants du problème (hyperparamètres) et peuvent donner des résultats relativement mauvais. Par exemple, les arbres de décision peuvent devoir diviser les données sans aucun gain de pureté pour atteindre de hauts gains de pureté à l'étape suivante. Il est donc courant de développer de grands arbres en utilisant le nombre d'exemples d'entraînement dans un nœud feuille comme critère d'arrêt. Pour éviter le sur-apprentissage, l'algorithme élaguera ensuite l'arbre résultant. Dans CART, le critère d'élagage $C_{pruning}(T)$ équilibre l'impureté et la complexité du modèle par régularisation. La variable régularisée est souvent le nombre de nœuds feuille $\vert T \vert$, comme ci-dessous :

$$C_{pruning}(T) = \sum^{\vert T \vert }_{v=1} I(T,v) + \lambda \vert T \vert$$

$\lambda$ est sélectionné via validation croisée et fait un compromis entre l'impureté et la complexité du modèle, pour un arbre donné $T$, avec les nœuds feuille $v=1...\vert T \vert$ utilisant la mesure d'impureté $I$.

**Variantes** : il existe différentes méthodes d'arbres de décision, qui diffèrent selon les points suivants :

* Critère de division ? Gini / Entropie.
* Technique pour réduire le sur-apprentissage ?
* Combien de variables peuvent être utilisées dans une division ?
* Création d'arbres binaires ?
* Gestion des valeurs manquantes ?
* Peuvent-ils gérer la régression ?
* Robustesse aux valeurs aberrantes ?

Variantes célèbres :
* **ID3** : première implémentation d'arbre de décision. Pas utilisé en pratique.
* **C4.5** : Amélioration par rapport à ID3 par le même développeur. Élagage basé sur l'erreur. Utilise l'entropie. Gère les valeurs manquantes. Sensible aux valeurs aberrantes. Peut créer des branches vides.
* **CART** : Utilise Gini. Élagage basé sur la complexité des coûts. Arbres binaires. Gère les valeurs manquantes. Gère la régression. Pas sensible aux valeurs aberrantes.
* **CHAID** : Trouve une variable de division en utilisant le test du Chi-carré pour tester la dépendance entre une variable et une réponse. Pas d'élagage. Semble mieux pour décrire les données, mais moins performant pour la prédiction.

Autres variantes : C5.0 (version suivante de C4.5, probablement moins utilisée car brevetée), MARS.

:information_source: <span class='resources'> Ressources </span> : Une étude comparative de [différentes méthodes d'arbres de décision](http://www.academia.edu/34100170/Comparative_Study_Id3_Cart_And_C4.5_Decision_Tree_Algorithm_A_Survey).
</div>
</details>
</div> 
<p></p>

<div>
<details>
<summary>Pseudocode et Complexité</summary>
<div markdown='1'>

* **Pseudocode**
La version simple d'un arbre de décision peut être écrite en quelques lignes de pseudocode Python :

```python
def construireArbre(X,Y):
    if critere_arret(X,Y) :
        # si arrêt, alors stocker la classe majoritaire
        arbre.classe = mode(X) 
        return Null

    impureteMin = infini
    meilleureDivision = None
    for j in caracteristiques:
        for T in seuils:
            if impurete(X,Y,j,T) < impureteMin:
                meilleureDivision = (j,T)
                impureteMin = impurete(X,Y,j,T) 

    X_gauche, Y_gauche, X_droite, Y_droite = diviser(X,Y,meilleureDivision)

    arbre.division = meilleureDivision # ajoute la division actuelle
    arbre.gauche = construireArbre(X_gauche, Y_gauche) # ajoute les divisions suivantes à gauche
    arbre.droite = construireArbre(X_droite, Y_droite) # ajoute les divisions suivantes à droite

return arbre

def predireUnArbre(arbre, xi):
    if arbre.classe n'est pas Null:
        return arbre.classe

    j, T = arbre.division
    if xi[j] >= T:
        return predireUnArbre(arbre.droite, xi)
    else:
        return predireUnArbre(arbre.gauche, xi)

def predireTousArbre(arbre, Xt):
    t, d = Xt.shape
    Yt = vecteur(d)
    for i in t:
        Yt[i] = predireUnArbre(arbre, Xt[i,:])

    return Yt
```

* **Complexité**
Je vais utiliser la notation suivante : $$M=profondeur$$ ; $$K=\#_{seuils}$$ ; $$N = \#_{train}$$ ; $$D = \#_{caractéristiques}$$ ; $$T = \#_{test}$$.

Commençons par réfléchir à la complexité de la construction de la première souche de décision (premier appel de fonction) :

* Dans une souche de décision, nous bouclons sur toutes les caractéristiques et les seuils $O(KD)$, puis nous calculons l'impureté. L'impureté dépend uniquement des probabilités de classe. Calculer les probabilités signifie boucler sur tous les $X$ et compter les $Y$ : $O(N)$. Avec ce pseudocode simple, la complexité temporelle pour construire une souche est donc $O(KDN)$. 
* En réalité, nous n'avons pas besoin de rechercher des seuils arbitraires, uniquement pour les valeurs uniques prises par au moins un exemple. *Par exemple*, pas besoin de tester $caractéristique_j>0.11$ et $caractéristique_j>0.12$ lorsque toutes les $caractéristique_j$ sont soit $0.10$ soit $0.80$. Remplaçons le nombre de seuils possibles $K$ par la taille du jeu d'entraînement $N$. $O(N^2D)$
* Actuellement, nous bouclons deux fois sur tous les $X$, une fois pour le seuil et une fois pour calculer l'impureté. Si les données étaient triées par la caractéristique actuelle, l'impureté pourrait simplement être mise à jour au fur et à mesure que nous parcourons les exemples. *Par exemple*, en considérant la règle $caractéristique_j>0.8$ après avoir déjà considéré $caractéristique_j>0.7$, nous n'avons pas besoin de recalculer toutes les probabilités de classe : nous pouvons simplement prendre les probabilités de $caractéristique_j>0.7$ et faire les ajustements en connaissant le nombre d'exemples avec $caractéristique_j==0.7$. Pour chaque caractéristique $j$, nous devons d'abord trier toutes les données $O(N\log(N))$, puis boucler une fois en $O(N)$, la finalité serait en $O(DN\log(N))$.

Nous avons maintenant la complexité d'une souche de décision. Vous pourriez penser que trouver la complexité de la construction d'un arbre serait de la multiplier par le nombre d'appels de fonction : Vrai ? Pas vraiment, ce serait une surestimation. En effet, à chaque appel de fonction, la taille des données d'entraînement $N$ aura diminué. L'intuition du résultat que nous recherchons est qu'à chaque niveau $l=1...M$, la somme des données d'entraînement dans chaque fonction est toujours $N$. Plusieurs fonctions travaillant en parallèle avec un sous-ensemble d'exemples prennent le même temps qu'une seule fonction avec l'ensemble d'entraînement complet $N$. La complexité à chaque niveau est donc toujours $O(DN\log(N))$, donc la complexité pour construire un arbre de profondeur $M$ est $O(MDN\log(N))$. Preuve que le travail à chaque niveau reste constant :

À chaque itération, le jeu de données est divisé en $\nu$ sous-ensembles de $k_i$ éléments et un ensemble de $n-\sum_{i=1}^{\nu} k_i$. À chaque niveau, le coût total serait donc (en utilisant les propriétés des logarithmes et le fait que $k_i \le N$ ) :

$$
\begin{align*}
coût &= O(k_1D\log(k_1)) + ... + O((N-\sum_{i=1}^{\nu} k_i)D\log(N-\sum_{i=1}^{\nu} k_i))\\
    &\le O(k_1D\log(N)) + ... + O((N-\sum_{i=1}^{\nu} k_i)D\log(N))\\
    &= O(((N-\sum_{i=1}^{\nu} k_i)+\sum_{i=1}^{\nu} k_i)D\log(N)) \\
    &= O(ND\log(N))   
\end{align*} 
$$

Le dernier ajustement possible que je vois est de tout trier une fois, de le stocker et d'utiliser simplement ces données pré-calculées à chaque niveau. La complexité d'entraînement finale est donc <span class='practiceText'> $O(MDN + ND\log(N))$ </span>.

La complexité temporelle pour faire des prédictions est simple : pour chaque $t$ exemple, parcourez une question à chaque niveau $M$. *C'est-à-dire* <span class='practiceText'> $O(MT)$ </span>.
</div>
</details>
</div> 
<p></p>

## Régression

Les 2 différences avec les arbres de décision pour la classification sont :
* **Quelle erreur minimiser pour une division optimale ?** Cela remplace la mesure d'impureté dans le contexte de classification. Une fonction d'erreur largement utilisée pour la régression est la somme des erreurs au carré. Nous n'utilisons pas l'erreur quadratique moyenne afin que la soustraction de l'erreur après et avant une division ait du sens. Somme des erreurs au carré pour la région $R$ :

$$Erreur = \sum_{x^{(n)} \in R} (y^{(n)} - \bar{y}_{R})^2$$

* **Que prédire pour une région donnée de l'espace ?** Dans le contexte de classification, nous avons prédit la modalité du sous-ensemble des données d'entraînement dans cet espace. Prendre la modalité n'a pas de sens pour une variable continue. Maintenant que nous avons défini une fonction d'erreur ci-dessus, nous aimerions prédire une valeur qui minimise cette fonction de somme des erreurs au carré. Cela correspond à la **valeur moyenne** de la région. Prédire la moyenne est intuitivement ce que nous aurions fait. 

Jetons un coup d'œil à un graphique simple pour mieux comprendre l'algorithme :

<div markdown="1">
![Construction des arbres de décision pour la régression](/images/blog/glossary-old/decision-tree-reg.gif){:width='477px' :height='327px'}
</div>

:x: En plus des inconvénients observés dans les arbres de décision pour la classification, les arbres de décision pour la régression souffrent du fait qu'ils prédisent une <span class='disadvantageText'> fonction non lisse </span>.