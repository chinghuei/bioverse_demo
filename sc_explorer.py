import dash
from dash import dcc, html
import plotly.express as px
import scanpy as sc
import pandas as pd
import requests

adata = sc.read_h5ad("data/processed.h5ad")

app = dash.Dash(__name__)

umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
umap_df["cell_id"] = adata.obs_names
umap_df["cell_type"] = (
    adata.obs["cell_type"] if "cell_type" in adata.obs else "unknown"
)

fig = px.scatter(
    umap_df,
    x="UMAP1",
    y="UMAP2",
    color="cell_type",
    hover_data=["cell_id"],
    custom_data=["cell_id"],
)
fig.update_layout(xaxis_scaleanchor="y")

app.layout = html.Div([
    html.H2("scRNA-seq Explorer"),
    dcc.Graph(
        id="umap",
        figure=fig,
        style={"height": "80vh", "width": "100%"},
        config={"responsive": True},
    ),
    html.Div(id="selected-cells")
])

@app.callback(
    dash.dependencies.Output("selected-cells", "children"),
    [dash.dependencies.Input("umap", "selectedData")]
)
def display_selected_cells(data):
    if data:
        points = []
        for p in data["points"]:
            cell_id = None
            if "customdata" in p and p["customdata"]:
                cell_id = p["customdata"][0]
            if cell_id is None:
                idx = p.get("pointIndex")
                if idx is not None and idx < len(umap_df):
                    cell_id = umap_df.iloc[idx]["cell_id"]
            if cell_id is not None:
                points.append(str(cell_id))

        try:
            requests.post(
                "http://127.0.0.1:5000/selection",
                json={"cell_ids": points},
                timeout=1,
            )
        except Exception as e:
            print("Failed to send selection", e)

        if points:
            return html.Pre("Selected cells:\n" + "\n".join(points))

    return "Select cells using lasso or box tool."

if __name__ == "__main__":
    app.run(debug=True, port=8050)
