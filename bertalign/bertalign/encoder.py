import os
import numpy as np
from sentence_transformers import SentenceTransformer
from bertalign.bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name_or_path):
        # Check if it's the default model name that needs to be loaded locally
        if model_name_or_path == "LaBSE":
            # Point to your local snapshot
            model_path = '/home/maria/.cache/huggingface/hub/models--sentence-transformers--LaBSE/snapshots'
            
            # Find the latest snapshot folder
            snapshot_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
            if snapshot_folders:
                model_path = os.path.join(model_path, snapshot_folders[0])
                print(f"Using local model from: {model_path}")
                self.model = SentenceTransformer(model_path, local_files_only=True)
            else:
                raise ValueError("No snapshot folders found in the specified path")
        else:
            # For other models, use the original loading method
            self.model = SentenceTransformer(model_name_or_path)
            
        self.model_name = model_name_or_path


    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs
