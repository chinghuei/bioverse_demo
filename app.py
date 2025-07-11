from flask import Flask, request, jsonify, session, render_template
import scanpy as sc
import anndata
import os
from data.bio_llm import load_models, answer_question

app = Flask(__name__)
app.secret_key = "demo_secret"

# === Setup upload/processed folders ===
UPLOAD_FOLDER = "data/uploads"
PROCESSED_FILE = "data/processed.h5ad"
CB_FOLDER = "data/cellbrowser"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CB_FOLDER, exist_ok=True)

# === Load default AnnData ===
adata = None
if os.path.exists(PROCESSED_FILE):
    adata = sc.read_h5ad(PROCESSED_FILE)

# Load multimodal models for inference
try:
    MODELS = load_models()
except Exception as e:
    MODELS = None
    print(f"Failed to load models: {e}")

# === Predict cell types using the loaded models ===
def run_model(selected_cells, question, history):
    if MODELS is None:
        return "Model not available"
    predictions = answer_question(selected_cells, question, MODELS)
    if len(predictions) == 1:
        return predictions[0]
    return "\n".join(predictions)

# Store cell selections coming from visualization tools
current_selection = []


@app.route("/selection", methods=["GET", "POST"])
def selection():
    """Store or retrieve the currently selected cell IDs."""
    global current_selection
    if request.method == "POST":
        data = request.get_json() or {}
        current_selection = data.get("cell_ids", [])
        return "ok"
    return jsonify({"cell_ids": current_selection})

@app.route("/")
def index():
    return render_template("layout.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    ids = data["cell_ids"]
    question = data["question"]
    history = session.get("chat", [])

    if adata is None:
        return "❌ No data loaded", 500

    selected = adata[adata.obs_names.isin(ids)]
    answer = run_model(selected, question, history)

    history.append({"q": question, "a": answer})
    session["chat"] = history

    return jsonify({"answer": answer, "history": history})

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "❌ No file part in the request", 400
    file = request.files['file']
    if file.filename == '':
        return "❌ No selected file", 400
    if not file.filename.endswith(".h5ad"):
        return "❌ Only .h5ad files are supported", 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Load and preprocess the new file
    global adata
    adata = sc.read_h5ad(save_path)

    try:
        # Basic preprocessing for visualization
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        # Save processed data for downstream tools
        adata.write(PROCESSED_FILE)
        adata.write(os.path.join(CB_FOLDER, "adata_cb.h5ad"))
    except Exception as e:
        return f"❌ Preprocessing failed: {str(e)}", 500

    return (
        f"✅ File '{file.filename}' uploaded and processed. "
        f"Saved to '{PROCESSED_FILE}'."
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
