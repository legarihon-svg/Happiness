import streamlit as st

st.set_page_config(page_title="Mon App Streamlit", page_icon="🚀")

st.title("🚀 Mon Application Streamlit")
st.write("Bienvenue dans votre application hébergée sur Hugging Face!")

# Interface interactive
nom = st.text_input("Entrez votre nom:")
if nom:
    st.success(f"Bonjour {nom}! 👋")

# Slider
valeur = st.slider("Sélectionnez une valeur:", 0, 100, 50)
st.write(f"Valeur sélectionnée: {valeur}")

# Graphique
if st.checkbox("Afficher un graphique"):
    import pandas as pd
    import numpy as np
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(chart_data)


