import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PARQUET_PATH = r"C:\Users\Arun\echs-bot\echs_segments_master.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"     # SentenceTransformer model
TS = time.strftime("%Y%m%d_%H%M%S")

EMB_OUT = f"embeddings.{TS}.npy"
IDX_OUT = f"echs_faiss.{TS}.index"
META_OUT = f"echs_faiss.{TS}.meta.json"

# -------------------------------------------------------------------
# LOAD SEGMENTS
# -------------------------------------------------------------------
print("\n==> Loading parquet file...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} rows.")

# Find the text column
TEXT_COL = None
for col in ("text", "segment_text", "chunk_text", "transcript"):
    if col in df.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    TEXT_COL = df.columns[0]

texts = df[TEXT_COL].fillna("").astype(str).tolist()

metas = df.to_dict(orient="records")

# -------------------------------------------------------------------
# COMPUTE EMBEDDINGS
# -------------------------------------------------------------------
print("\n==> Computing embeddings... (this may take 1–4 minutes)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)

embs = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    convert_to_numpy=True
).astype("float32")

print("Embeddings shape:", embs.shape)
np.save(EMB_OUT, embs)
print(f"Saved raw embeddings -> {EMB_OUT}")

# -------------------------------------------------------------------
# NORMALIZE (for cosine similarity)
# -------------------------------------------------------------------
print("\n==> Normalizing embeddings...")
norms = np.linalg.norm(embs, axis=1, keepdims=True)
norms[norms == 0] = 1.0
embs_norm = (embs / norms).astype("float32")

# -------------------------------------------------------------------
# BUILD FAISS INDEX
# -------------------------------------------------------------------
print("\n==> Building FAISS index (IndexFlatIP)...")
import faiss

dim = embs_norm.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs_norm)

faiss.write_index(index, IDX_OUT)
print(f"Saved FAISS index -> {IDX_OUT}")

# -------------------------------------------------------------------
# SAVE METADATA
# -------------------------------------------------------------------
with open(META_OUT, "w", encoding="utf8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)
print(f"Saved metadata -> {META_OUT}")

# -------------------------------------------------------------------
# VALIDATION
# -------------------------------------------------------------------
print("\n==> Validating new index...")
D, I = index.search(embs_norm[0:1], 5)
print("Self-match score (should be ~1.0):", float(D[0][0]))
print("Index ntotal:", index.ntotal)

print("\n==> DONE.")
print("Files created:")
print(" ", EMB_OUT)
print(" ", IDX_OUT)
print(" ", META_OUT)
print("\nYou may now swap these fresh files into app.py if satisfied.")
