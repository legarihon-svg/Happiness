# Connexion d'une BD PostgreSQL à Power BI Desktop 🚀

## Pré-requis

👉 Installer le connecteur PostgreSQL pour Power BI Desktop :
**Npgsql-4.0.17.msi** (présent dans ce dossier)

📖 Documentation officielle : [https://learn.microsoft.com/fr-fr/power-query/connectors/postgresql](https://learn.microsoft.com/fr-fr/power-query/connectors/postgresql)

---

## Étapes dans Power BI Desktop

### 1. Obtenir des données d'une autre source

Dans l'accueil de Power BI Desktop, cliquer sur **"Obtenir des données d'une autre source"**.

---

### 2. Se connecter à une base de données PostgreSQL

Dans la liste des connecteurs, sélectionner **Base de données PostgreSQL**.

---

### 3. Renseigner les informations du serveur

| Champ | Valeur |
|---|---|
| **Serveur** | `ep-crimson-rain-agcmctv6-pooler.c-2.eu-central-1.aws.neon.tech` |
| **Base de données** | `neondb` |

---

### 4. Renseigner les identifiants

| Champ | Valeur |
|---|---|
| **Nom d'utilisateur** | `neondb_owner` |
| **Mot de passe** | `xxxxxxxxxxxxxxxxxxxxxxxx` |

---

### 5. Choisir la table et charger

Sélectionner la table **`public_world_happiness_pbi`** — il s'agit de la vue de la base de données adaptée pour Power BI.

Cliquer sur **→ Charger**.
