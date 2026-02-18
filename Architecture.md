# 📘 **Architecture de Données – World Happiness 2025**  
### *Du fichier brut à l’analyse prédictive : pipeline ETL, stockage serverless et visualisation moderne*

---

## 1. 🎯 **Objectif du Projet**

Ce projet vise à transformer des fichiers statiques (Excel et CSV) issus du *World Happiness Report* et de la *Banque Mondiale* en un **actif de données dynamique**, centralisé dans une base PostgreSQL serverless et exploitable en temps réel par :

- **Power BI** pour l’analyse descriptive  
- **Streamlit** pour l’exploration avancée et le Machine Learning  

---

## 2. 🏗️ **Vue d’Ensemble de l’Architecture**

L’architecture repose sur trois zones fonctionnelles :

### **Zone 01 — Ingestion**
- Extraction de fichiers hétérogènes : `.xlsx` (bonheur subjectif) et `.csv` (indicateurs économiques).
- Harmonisation des structures avant fusion.

> « Le pipeline débute par l'extraction de deux formats de fichiers distincts. Le défi technique réside dans l'harmonisation de ces structures disparates. »

### **Zone 02 — Stockage**
- Chargement dans PostgreSQL serverless (Neon).
- Séparation compute/storage pour scalabilité automatique.

> « Chargement : Centralisation dans un entrepôt PostgreSQL Serverless. »

### **Zone 03 — Exploitation**
- Power BI pour la BI.
- Streamlit pour l’analytique avancée et les prédictions.

> « Visualisation : Double interface pour l'analytique (Power BI) et le Machine Learning (Streamlit). »

---

## 3. 🔧 **Pipeline ETL (Python)**

### 3.1 Extraction
Les sources sont :

- `WHR25_Data_Figure_2.1.xlsx` (scores de bonheur)
- `world_bank_data_2025.csv` (indicateurs économiques)

### 3.2 Transformation
La fusion repose sur une **clé composite** :

```python
pd.merge(
    left=happiness_df,
    right=economy_df,
    on=['country', 'year'],
    how='inner'
)
```

> « L’unification des données repose sur une clé composite : le pays et l’année. »

### 3.3 Nettoyage
Le protocole de purification inclut :

- suppression de colonnes inutiles  
- normalisation des noms  
- typage strict  

> « Avant le chargement, un protocole de nettoyage strict est appliqué. »

---

## 4. 🗄️ **Stockage : PostgreSQL Serverless (Neon)**

### 4.1 Infrastructure
- Base PostgreSQL serverless
- Scalabilité automatique
- Séparation compute/storage

> « Ce choix technique permet une scalabilité automatique et une séparation entre le calcul et le stockage. »

### 4.2 Table cible
`world_happiness`

### 4.3 Vue sémantique
`world_happiness_pbi`

Rôle de la vue :

- conversion des types  
- abstraction de la couche physique  

> « Nous ne connectons jamais les outils de BI directement à la table brute. »

---

## 5. 📊 **Consommation des Données**

### 5.1 Power BI — Business Intelligence
Power BI se connecte **uniquement à la vue SQL**.

Étapes clés :

1. Connexion via `world_happiness_pbi`
2. Typage strict dans Power Query
3. Création de dashboards interactifs

> « Le typage final est forcé dans Power Query pour garantir la précision des agrégations. »

### 5.2 Streamlit — Machine Learning & Apps
L’application Streamlit hébergée sur HuggingFace :

- se connecte directement à Neon  
- permet l’exploration interactive  
- intègre des modèles ML en temps réel  

> « Elle permet d'intégrer des modèles de Machine Learning directement sur les données en temps réel. »

---

## 6. 🔐 **Gouvernance & Qualité**

### 6.1 Intégrité
- Fusion stricte sur clé `(country, year)`
- Nettoyage systématique avant stockage

### 6.2 Performance
- Architecture serverless
- Vue optimisée pour la consommation

### 6.3 Flexibilité
- Une seule source de vérité pour BI + ML

---

## 7. 📌 **Conclusion**

Cette architecture propose un pipeline moderne, robuste et scalable, permettant :

- une ingestion fiable  
- une transformation contrôlée  
- un stockage optimisé  
- une exploitation polyvalente  

