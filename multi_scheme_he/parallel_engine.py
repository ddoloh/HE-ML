import time
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class CPUParallelInferenceEngine:
    """
    Improvement 2: Multi-Threaded Parallel CPU Inference Engine.
    Leverages thread pool executor over CPU cores to accelerate batch inference throughput.
    """
    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = min(8, os.cpu_count() or 4)
        self.max_workers = max_workers

    def run_parallel_batch(self, model_predict_fn, batch_inputs):
        """
        Splits batch_inputs into parallel chunks across CPU worker threads.
        """
        t0 = time.perf_counter()
        
        n_samples = len(batch_inputs)
        chunk_size = max(1, n_samples // self.max_workers)
        chunks = [batch_inputs[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(model_predict_fn, chunk) for chunk in chunks]
            for f in futures:
                res = f.result()
                if isinstance(res, np.ndarray):
                    results.extend(res.tolist())
                elif isinstance(res, list):
                    results.extend(res)
                else:
                    results.append(res)
                    
        t1 = time.perf_counter()
        total_time = t1 - t0
        throughput = n_samples / max(total_time, 1e-6)
        
        return {
            "n_samples": n_samples,
            "max_workers": self.max_workers,
            "total_time_ms": round(total_time * 1000.0, 3),
            "parallel_throughput": round(throughput, 1),
            "results": results
        }
