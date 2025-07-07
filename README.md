# Multimodal scRNA-seq Demo

This is a demo for exploring scRNA-seq data using two interactive visualization tools and asking natural language questions to a multimodal model (LLaVA-style). It supports:

- Loading and preprocessing `.h5ad` data (AnnData)
- Visualizing UMAP and metadata via:
  - UCSC Cell Browser (interactive selection)
  - scExplorer (Dash-based frontend)
- Asking questions like “What cell types are these?” on selected cells
- Receiving placeholder model responses (e.g., from a LLaVA-style backend)

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
