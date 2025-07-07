import scanpy as sc
import os

os.makedirs("data/cellbrowser", exist_ok=True)

adata = sc.read_h5ad("data/demo.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.write("data/processed.h5ad")

# Save a version compatible with UCSC Cell Browser
adata.write("data/cellbrowser/adata_cb.h5ad")

# After this, run:
#   cbBuild --matrix data/cellbrowser/adata_cb.h5ad --name demo
# convert .h5ad for UCSC Cell Browser and scExplorer
