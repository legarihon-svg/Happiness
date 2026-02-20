😊 World Happiness

📊 Vue d'ensemble
Cette application combine :

Données historiques : World Happiness Report + World Bank (2011–2024)
Machine Learning : XGBoost (12 modèles) + Random Forest (importance des variables)
Projections : Prédictions macroéconomiques jusqu'en 2030

3 pages interactives

📈 Prédit vs Réel — Comparaison historique/projection par pays avec heatmap des résidus
🗺️ Carte mondiale — Choroplèthe interactive (valeurs absolues ou variations annuelles)
🔬 Variables macro — Importance Random Forest + corrélations 2025–2030 + scatter plots


🐳 Déploiement sur Hugging Face Spaces (Docker)
Architecture
world-happiness-predictor/
├── app.py              # Application Streamlit
├── requirements.txt    # Dépendances Python
├── Dockerfile          # Configuration Docker
└── README.md           # Cette doc


Créer un Space sur huggingface.co/spaces

Type : Docker
SDK : Streamlit (optionnel, mais aide l'UI)


Pousser les fichiers

bash   git clone https://huggingface.co/spaces/YOUR_USERNAME/world-happiness
   cd world-happiness
   cp app.py requirements.txt Dockerfile README.md .
   git add .
   git commit -m "Initial commit"
   git push


Attendre le build (2–5 min)

Docker build → pull des données → démarrage de Streamlit
L'app est accessible à https://huggingface.co/spaces/YOUR_USERNAME/world-happiness
