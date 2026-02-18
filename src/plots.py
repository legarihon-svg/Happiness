import matplotlib.pyplot as plt
import plotly.express as px

def plot_importances(df, title):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)  # taille réelle + DPI

    ax.barh(df["feature"], df["importance"], color="#052251")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Importance", fontsize=8)
    ax.set_ylabel("Variable", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig

import plotly.express as px

def plot_world_map(df, column, title):
    fig = px.choropleth(
        df,
        locations="Code ISO",
        color=column,
        hover_name="Pays",
        color_continuous_scale="Viridis",
        projection="orthographic",
        title=title,
    )

    fig.update_geos(
        showocean=True,
        oceancolor="rgb(0, 60, 150)",
        landcolor="rgb(230, 230, 230)",
        showland=True,
        bgcolor="black"
    )

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
        height=700
    )

    return fig   # ← IMPORTANT : ne jamais enlever ça