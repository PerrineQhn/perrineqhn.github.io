---
layout: archive
title: "Portfolio"
permalink: /portfolio/
author_profile: true
---

Bienvenue dans mon portfolio ! Vous trouverez ici un aperçu de mes projets en traitement automatique du langage, en analyse de données, en intelligence artificielle, et en développement d’applications interactives.

{% for item in site.portfolio %}
  {% assign mois = item.date | date: "%m" %}
  {% assign mois_fr = "" %}
  {% case mois %}
    {% when "01" %}{% assign mois_fr = "janvier" %}
    {% when "02" %}{% assign mois_fr = "février" %}
    {% when "03" %}{% assign mois_fr = "mars" %}
    {% when "04" %}{% assign mois_fr = "avril" %}
    {% when "05" %}{% assign mois_fr = "mai" %}
    {% when "06" %}{% assign mois_fr = "juin" %}
    {% when "07" %}{% assign mois_fr = "juillet" %}
    {% when "08" %}{% assign mois_fr = "août" %}
    {% when "09" %}{% assign mois_fr = "septembre" %}
    {% when "10" %}{% assign mois_fr = "octobre" %}
    {% when "11" %}{% assign mois_fr = "novembre" %}
    {% when "12" %}{% assign mois_fr = "décembre" %}
  {% endcase %}
  
  ## [{{ item.title }}]({{ item.url }})
  
  **Date :** {{ item.date | date: "%d" }} {{ mois_fr }} {{ item.date | date: "%Y" }}
  
  {{ item.excerpt }}

  [Lire plus]({{ item.url }})

  ---
{% endfor %}