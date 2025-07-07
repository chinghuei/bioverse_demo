from flask import Flask, request, jsonify, session, render_template
import scanpy as sc
import anndata
import os

app = Flask(__name__)
app.secret_key = "demo_secret"

# Load AnnData (assuming it’s preprocessed with UMAP)
adata = sc.read_h5ad("data/processed.h5ad")

# === Placeholder for LLaVA-style multimodal model ===
def run_model(selected_cells, question, history):
    return f"(Placeholder) You asked: '{question}' about {len(selected_cells)} selected cells."

@app.route("/")
def index():
    return render_template("layout.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    ids = data["cell_ids"]
    question = data["question"]
    history = session.get("chat", [])

    selected = adata[adata.obs_names.isin(ids)]
    answer = run_model(selected, question, history)

    history.append({"q": question, "a": answer})
    session["chat"] = history

    return jsonify({"answer": answer, "history": history})

if __name__ == "__main__":
    app.run(debug=True)
# Flask app (placeholder content)
