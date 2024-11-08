---
layout: archive
title: "Portfolio"
permalink: /portfolio/
author_profile: true
---

# Portfolio

Bienvenue dans mon portfolio ! Vous trouverez ici un aperçu de mes projets en traitement automatique du langage, en analyse de données, en intelligence artificielle, et en développement d’applications interactives.

{% for item in site.portfolio %}

## [{{ item.title }}]({{ item.url }})

**Date :** {{ item.date | date: "%d %B %Y" | localize: 'fr' }}

{{ item.excerpt }}

[Lire plus]({{ item.url }})

---

{% endfor %}