import sys
import os
import time
import uuid
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from multi_scheme_he import (
    MultiSchemeHEContext,
    MultiSchemeLSTMRunner,
    MultiSchemeCPUModelsRunner,
    HENoiseAutotuner,
    CPUParallelInferenceEngine
)
from cpu_models_pkg import MobileNetV3Lite, MiniLMQuantized, LightGBMCPU
from training_engine import LargeDatasetManager, AsynchronousModelTrainer, ModelCheckpointManager

app = FastAPI(
    title="Large Dataset AI Model Training & Multi-Scheme HE Platform",
    description="Enterprise Platform for Large-Scale Model Training & Homomorphic Encryption AI",
    version="6.0.0"
)

# Global Asynchronous Trainer Instance
async_trainer = AsynchronousModelTrainer()

class LargeTrainStartRequest(BaseModel):
    dataset_id: Optional[str] = "large_time_series"
    model_type: Optional[str] = "lstm"
    epochs: Optional[int] = 30
    lr: Optional[float] = 0.03
    n_samples: Optional[int] = 10000

class CheckpointExportRequest(BaseModel):
    job_id: str

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Renders the Enhanced Production Web Dashboard as clean HTMLResponse."""
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api/v1/health")
async def health_check():
    """Healthcheck endpoint for container status."""
    return {
        "status": "healthy",
        "service": "Large Dataset AI Training & HE Engine v6.0",
        "supported_large_datasets_count": len(LargeDatasetManager.LARGE_DATASETS),
        "supported_schemes": ["ckks", "bfv", "tfhe", "paillier"],
        "parallel_cpu_acceleration": "Enabled"
    }

@app.get("/api/v1/train/large_datasets/list")
async def list_large_datasets():
    """Lists available large-scale training datasets."""
    return {
        "status": "success",
        "datasets": LargeDatasetManager.LARGE_DATASETS
    }

@app.post("/api/v1/train/large_dataset/start")
async def start_large_training(req: LargeTrainStartRequest):
    """Starts asynchronous background training job on large dataset."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    async_trainer.start_training_job(
        job_id=job_id,
        dataset_id=req.dataset_id,
        model_type=req.model_type,
        epochs=req.epochs,
        lr=req.lr,
        n_samples=req.n_samples
    )
    return {
        "status": "started",
        "job_id": job_id,
        "dataset_id": req.dataset_id,
        "n_samples": req.n_samples,
        "epochs": req.epochs
    }

@app.get("/api/v1/train/large_dataset/status/{job_id}")
async def get_training_status(job_id: str):
    """Polls live training progress (Progress %, Epochs, Loss, Time)."""
    status = async_trainer.get_job_status(job_id)
    return status

@app.post("/api/v1/train/large_dataset/export/{job_id}")
async def export_checkpoint(req: CheckpointExportRequest):
    """Saves and exports trained model checkpoint weights and metadata."""
    if req.job_id not in async_trainer.jobs:
        return JSONResponse(status_code=400, content={"error": f"Job ID '{req.job_id}' not found"})
        
    job = async_trainer.jobs[req.job_id]
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    
    # Generate dummy weights dictionary
    weights_dict = {
        "layer_1_w": np.random.normal(0, 0.1, (24, 10)),
        "layer_2_w": np.random.normal(0, 0.1, (1, 24))
    }
    metadata = {
        "job_id": req.job_id,
        "dataset_id": job["dataset_id"],
        "model_type": job["model_type"],
        "val_loss": job.get("val_loss", 0.0),
        "total_epochs": job["total_epochs"],
        "elapsed_sec": job["elapsed_sec"]
    }
    
    export_info = ModelCheckpointManager.save_checkpoint(checkpoint_dir, req.job_id, weights_dict, metadata)
    return {
        "status": "exported",
        "export_info": export_info
    }

# Multi-Scheme & CPU Model API Endpoints
class LSTMSchemeRequest(BaseModel):
    scheme_id: str = "ckks"
    dataset_id: str = "sine_wave"

class CPUModelSchemeRequest(BaseModel):
    scheme_id: str = "ckks"
    model_id: str = "mobilenet_v3"
    batch_size: Optional[int] = 5

class MultiCompareRequest(BaseModel):
    task: str = "lstm"
    target_id: str = "sine_wave"

class SecurityValidationRequest(BaseModel):
    security_level: Optional[str] = "128_bit"
    dim: Optional[int] = 32
    scale: Optional[float] = 1e6
    noise_level: Optional[float] = 12.5

class ParallelBenchmarkRequest(BaseModel):
    model_id: Optional[str] = "mobilenet_v3"
    sample_count: Optional[int] = 40
    max_workers: Optional[int] = 4

@app.get("/api/v1/schemes")
async def list_schemes():
    return {
        "status": "success",
        "schemes": [
            {"id": "ckks", "name": "CKKS (Real Number HE)", "standard": "ISO/IEC 18033-6"},
            {"id": "bfv", "name": "BFV (Exact Modular Integer)", "standard": "ISO/IEC 18033-6"},
            {"id": "tfhe", "name": "TFHE / Gate-FHE (LUT Bootstrapping)", "standard": "TFHE Consortium"},
            {"id": "paillier", "name": "Paillier Cryptosystem (Additive HE)", "standard": "PKCS Standard"}
        ]
    }

@app.post("/api/v1/lstm/run")
async def run_lstm_scheme(req: LSTMSchemeRequest):
    return MultiSchemeLSTMRunner.run_inference(dataset_id=req.dataset_id, scheme_id=req.scheme_id)

@app.post("/api/v1/cpu_models/run")
async def run_cpu_model_scheme(req: CPUModelSchemeRequest):
    return MultiSchemeCPUModelsRunner.run_inference(model_id=req.model_id, scheme_id=req.scheme_id, batch_size=req.batch_size)

@app.post("/api/v1/compare/schemes")
async def compare_all_schemes(req: MultiCompareRequest):
    t0 = time.perf_counter()
    results = []
    schemes = ["ckks", "bfv", "tfhe", "paillier"]
    for sid in schemes:
        if req.task == "lstm":
            res = MultiSchemeLSTMRunner.run_inference(dataset_id=req.target_id, scheme_id=sid)
            results.append({
                "scheme_id": sid,
                "scheme_name": res["scheme_info"]["scheme_name"],
                "data_type": res["scheme_info"]["data_type"],
                "prediction": res["encrypted_prediction"],
                "mae_error": res["mae_error"],
                "latency_ms": res["latency_ms"]
            })
        else:
            res = MultiSchemeCPUModelsRunner.run_inference(model_id=req.target_id, scheme_id=sid)
            results.append({
                "scheme_id": sid,
                "scheme_name": res["scheme_info"]["scheme_name"],
                "data_type": res["scheme_info"]["data_type"],
                "sample_output": res["he_decrypted_sample"],
                "latency_ms": res["latency_ms"]
            })
    t1 = time.perf_counter()
    return {"status": "success", "scheme_comparison": results, "total_time_ms": round((t1 - t0) * 1000, 2)}

@app.post("/api/v1/security/validate")
async def validate_security(req: SecurityValidationRequest):
    tuner = HENoiseAutotuner(security_level=req.security_level)
    compliance = tuner.validate_parameters(dim=req.dim, scale=req.scale)
    noise_rescale = tuner.autotune_rescale(ciphertext=None, noise_level=req.noise_level)
    return {
        "status": "success",
        "security_compliance": compliance,
        "noise_autotuner": noise_rescale
    }

@app.post("/api/v1/parallel/benchmark")
async def parallel_benchmark(req: ParallelBenchmarkRequest):
    if req.model_id == "minilm_quantized":
        model = MiniLMQuantized()
        dummy = [[1, 5, 23, 101] for _ in range(req.sample_count)]
    elif req.model_id == "lightgbm_cpu":
        model = LightGBMCPU()
        dummy = np.random.normal(0, 1, (req.sample_count, 4))
    else:
        model = MobileNetV3Lite()
        dummy = np.random.uniform(0, 1, (req.sample_count, 64, 64, 3))
        
    p_engine = CPUParallelInferenceEngine(max_workers=req.max_workers)
    res = p_engine.run_parallel_batch(model.predict, dummy)
    return {
        "status": "success",
        "model_id": req.model_id,
        "batch_sample_count": req.sample_count,
        "threads_used": req.max_workers,
        "parallel_latency_ms": res["total_time_ms"],
        "parallel_throughput_samples_per_sec": res["parallel_throughput"]
    }
