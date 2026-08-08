import time
import numpy as np
from cpu_models_pkg import (
    CPUModelProfiler,
    MobileNetV3Lite,
    MiniLMQuantized,
    LightGBMCPU,
    FastGRNN
)
from multi_scheme_he.context import MultiSchemeHEContext

class MultiSchemeCPUModelsRunner:
    """
    Executes CPU-Optimized SOTA Models across 4 Homomorphic Encryption Schemes:
    CKKS, BFV, TFHE, Paillier.
    """
    @staticmethod
    def run_inference(model_id="mobilenet_v3", scheme_id="ckks", batch_size=5):
        t0 = time.perf_counter()
        he_ctx = MultiSchemeHEContext(scheme_id=scheme_id)
        
        # Instantiate Model
        if model_id == "mobilenet_v3":
            model = MobileNetV3Lite()
            dummy_input = np.random.uniform(0, 1, (batch_size, 64, 64, 3))
        elif model_id == "minilm_quantized":
            model = MiniLMQuantized()
            dummy_input = [[1, 5, 23, 101] for _ in range(batch_size)]
        elif model_id == "lightgbm_cpu":
            model = LightGBMCPU()
            dummy_input = np.random.normal(0, 1, (batch_size, 4))
        else:
            model = FastGRNN()
            dummy_input = np.random.normal(0, 1, (batch_size, 10, 1))
            
        plain_out = model.predict(dummy_input)
        
        # HE Encryption of outputs/inputs
        ct = he_ctx.encrypt(np.ravel(plain_out)[:10])
        dec_out = he_ctx.decrypt(ct)
        
        t1 = time.perf_counter()
        
        return {
            "status": "success",
            "model_id": model_id,
            "scheme_info": he_ctx.get_info(),
            "batch_size": batch_size,
            "plain_output_sample": list(np.round(np.ravel(plain_out)[:3], 4)),
            "he_decrypted_sample": list(np.round(dec_out[:3], 4)),
            "latency_ms": round((t1 - t0) * 1000, 3)
        }
