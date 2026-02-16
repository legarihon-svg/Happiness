FROM python:3.9-slim

WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .

# Exposer le port pour Streamlit
EXPOSE 7860

# Créer un utilisateur non-root
RUN useradd -m -u 1000 user
USER user

# Variable d'environnement pour le home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copier les fichiers dans le home de l'utilisateur
COPY --chown=user app.py .

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]