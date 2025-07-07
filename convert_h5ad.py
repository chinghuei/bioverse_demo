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
#   # Step 1: Convert .h5ad to Cell Browser format
#   cbImportScanpy -i data/cellbrowser/adata_cb.h5ad -o data/cellbrowser/demo
#
#   # Step 2: Build the Cell Browser AND serve it locally on port 8080
#   cbBuild -i data/cellbrowser/demo/cellbrowser.conf -o data/cellbrowser/ -p 8080

# convert .h5ad for UCSC Cell Browser and scExplorer
