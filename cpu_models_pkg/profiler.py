import time
import os
import resource
import numpy as np

class CPUModelProfiler:
    """
    Performance and Resource Profiler for CPU-Optimized Lightweight Models.
    Measures execution latency, memory footprint, throughput, and CPU efficiency score.
    """
    @staticmethod
    def profile_inference(model_fn, input_data, n_runs=10):
        """
        Executes model_fn(input_data) n_runs times to benchmark CPU latency and memory usage.
        """
        # Warmup run
        _ = model_fn(input_data)
        
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 # MB
        
        t0 = time.perf_counter()
        results = None
        for _ in range(n_runs):
            results = model_fn(input_data)
        t1 = time.perf_counter()
        
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 # MB
        
        total_time = t1 - t0
        avg_latency_ms = (total_time / n_runs) * 1000.0
        n_samples = len(input_data) if hasattr(input_data, '__len__') else 1
        throughput = (n_samples * n_runs) / max(total_time, 1e-6)
        mem_used_mb = max(0.05, mem_after - mem_before)
        
        efficiency_score = (throughput / (mem_used_mb + 1.0)) * 0.1
        
        return {
            "avg_latency_ms": round(avg_latency_ms, 3),
            "throughput_samples_per_sec": round(throughput, 1),
            "est_memory_mb": round(mem_used_mb, 2),
            "cpu_efficiency_score": round(efficiency_score, 2),
            "results": results
        }
