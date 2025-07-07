import dash
from dash import dcc, html
import plotly.express as px
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("data/processed.h5ad")

app = dash.Dash(__name__)

umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
umap_df["cell_id"] = adata.obs_names
umap_df["cell_type"] = adata.obs["cell_type"] if "cell_type" in adata.obs else "unknown"

fig = px.scatter(umap_df, x="UMAP1", y="UMAP2", color="cell_type", hover_data=["cell_id"])

app.layout = html.Div([
    html.H2("scExplorer (Dash)"),
    dcc.Graph(id="umap", figure=fig),
    html.Div(id="selected-cells")
])

@app.callback(
    dash.dependencies.Output("selected-cells", "children"),
    [dash.dependencies.Input("umap", "selectedData")]
)
def display_selected_cells(data):
    if data:
        points = [p["customdata"][0] for p in data["points"]]
        return html.Pre("Selected cells:\n" + "\n".join(points))
    return "Select cells using lasso or box tool."

if __name__ == "__main__":
    app.run_server(port=8050)
