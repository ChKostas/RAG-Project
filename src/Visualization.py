import numpy as np
from pathlib import Path
from chromadb import PersistentClient
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

try:
    pio.renderers.default = "vscode"
except Exception:
    pio.renderers.default = "notebook_connected"


BASE_DIR = Path.cwd()
DB_PATH = str(BASE_DIR / "preprocessed_db")
COLLECTION_NAME = "docs"

client = PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

data = collection.get(include=["embeddings", "metadatas", "documents"])

vectors = np.array(data["embeddings"])
metadatas = data["metadatas"]
documents = data["documents"]

file_names = [str(meta.get("file_name", "unknown")).strip() for meta in metadatas]
doc_types = [str(meta.get("type", "unknown")).strip().lower() for meta in metadatas]
chunk_ids = [str(meta.get("chunk_id", "unknown")).strip() for meta in metadatas]

print("Unique files:", sorted(set(file_names)))
print("Total files:", len(set(file_names)))

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=10,
    init="pca",
    learning_rate="auto"
)

reduced_vectors = tsne.fit_transform(vectors)

unique_files = sorted(set(file_names))

palette = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Light24
)

color_map = {file_name: palette[i % len(palette)] for i, file_name in enumerate(unique_files)}

fig = go.Figure()

for file_name in unique_files:
    xs = []
    ys = []
    texts = []

    for i, current_file in enumerate(file_names):
        if current_file == file_name:
            xs.append(reduced_vectors[i, 0])
            ys.append(reduced_vectors[i, 1])
            texts.append(
                f"File: {file_names[i]}<br>"
                f"Type: {doc_types[i]}<br>"
                f"Chunk ID: {chunk_ids[i]}<br>"
                f"Text: {documents[i][:220]}..."
            )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name=file_name,
            marker=dict(
                size=7,
                color=color_map[file_name],
                opacity=0.8
            ),
            text=texts,
            hoverinfo="text"
        )
    )

fig.update_layout(
    title="2D Chroma Vector Store Visualization",
    xaxis_title="t-SNE Dimension 1",
    yaxis_title="t-SNE Dimension 2",
    width=1100,
    height=750
)

fig.write_html("tsne_plot.html", auto_open=True)