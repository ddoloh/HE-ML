import os
import json
import numpy as np

class ModelCheckpointManager:
    """
    Saves and exports trained model weights and metadata to disk.
    """
    @staticmethod
    def save_checkpoint(checkpoint_dir, model_name, weights_dict, metadata):
        os.makedirs(checkpoint_dir, exist_ok=True)
        weights_path = os.path.join(checkpoint_dir, f"{model_name}_weights.npz")
        meta_path = os.path.join(checkpoint_dir, f"{model_name}_meta.json")
        
        np.savez(weights_path, **weights_dict)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "weights_path": weights_path,
            "meta_path": meta_path,
            "size_kb": round(os.path.getsize(weights_path) / 1024.0, 2)
        }
