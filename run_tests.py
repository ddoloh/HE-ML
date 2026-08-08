import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from he_ml import (
    HEContext,
    EncryptedVector,
    EncryptedMatrix,
    encrypted_sum,
    encrypted_mean,
    encrypted_variance,
    encrypted_covariance,
    EncryptedLinearRegression,
    EncryptedLogisticRegression,
    EncryptedDenseLayer,
    EncryptedSequentialNetwork,
    EncryptedKMeans
)

def format_banner(title):
    return f"\n{'='*60}\n  {title}\n{'='*60}"

def run_all_tests():
    print(format_banner("HOMOMORPHIC ENCRYPTION MACHINE LEARNING (HE-ML) TEST SUITE"))
    
    # Setup Encryption Context
    ctx = HEContext(dim=32, scale=1e6, seed=42)
    print(f"[*] Encryption Context Initialized (Vector Dim: {ctx.dim}, Scale: {ctx.scale:.0e})")

    # -------------------------------------------------------------
    # Test 1: Basic Encryption & Homomorphic Arithmetic
    # -------------------------------------------------------------
    print(format_banner("1. Basic Encryption & Homomorphic Arithmetic Test"))
    a = np.array([1.5, 2.5, 3.0, 4.0])
    b = np.array([0.5, 1.0, -1.5, 2.0])
    
    a_enc = EncryptedVector.encrypt(ctx, a)
    b_enc = EncryptedVector.encrypt(ctx, b)
    
    add_dec = (a_enc + b_enc).decrypt()
    sub_dec = (a_enc - b_enc).decrypt()
    mul_dec = (a_enc * b_enc).decrypt()
    dot_dec = a_enc.dot(b_enc).decrypt()[0]
    
    print(f"Plaintext A:          {a}")
    print(f"Plaintext B:          {b}")
    print(f"HE Addition:          {np.round(add_dec, 4)}  | True: {a+b}")
    print(f"HE Subtraction:       {np.round(sub_dec, 4)}  | True: {a-b}")
    print(f"HE Element-wise Mul:  {np.round(mul_dec, 4)}  | True: {a*b}")
    print(f"HE Dot Product:       {dot_dec:.4f}           | True: {np.dot(a, b):.4f}")

    # -------------------------------------------------------------
    # Test 2: Statistical Computations on Encrypted Data
    # -------------------------------------------------------------
    print(format_banner("2. Encrypted Data Statistical Analytics Test"))
    x_data = np.array([10.0, 12.0, 15.0, 18.0, 20.0, 25.0])
    y_data = np.array([5.0, 8.0, 9.0, 12.0, 15.0, 18.0])
    
    x_enc = EncryptedVector.encrypt(ctx, x_data)
    y_enc = EncryptedVector.encrypt(ctx, y_data)
    
    sum_he = encrypted_sum(x_enc).decrypt()[0]
    mean_he = encrypted_mean(x_enc).decrypt()[0]
    var_he = encrypted_variance(x_enc).decrypt()[0]
    cov_he = encrypted_covariance(x_enc, y_enc).decrypt()[0]
    
    print(f"Raw Input X:          {x_data}")
    print(f"HE Sum:               {sum_he:.4f}  | True: {np.sum(x_data):.4f}")
    print(f"HE Mean:              {mean_he:.4f}  | True: {np.mean(x_data):.4f}")
    print(f"HE Variance:          {var_he:.4f}  | True: {np.var(x_data):.4f}")
    print(f"HE Covariance(X, Y):  {cov_he:.4f}  | True: {np.cov(x_data, y_data, bias=True)[0, 1]:.4f}")

    # -------------------------------------------------------------
    # Test 3: Encrypted Linear Regression Model
    # -------------------------------------------------------------
    print(format_banner("3. Encrypted Linear Regression Model Test"))
    # Target function: y = 2.5 * x1 - 1.5 * x2 + 3.0
    X_raw = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 5.0]
    ])
    y_raw = 2.5 * X_raw[:, 0] - 1.5 * X_raw[:, 1] + 3.0
    
    lin_reg = EncryptedLinearRegression(ctx, n_features=2)
    lin_reg.fit(X_raw, y_raw, epochs=100, lr=0.02)
    
    X_enc = EncryptedMatrix.encrypt(ctx, X_raw)
    y_pred_enc = lin_reg.predict(X_enc)
    y_pred_dec = y_pred_enc.decrypt()
    
    print(f"True Target Y:        {y_raw}")
    print(f"Encrypted Pred Y:     {np.round(y_pred_dec, 4)}")
    print(f"Model Learned W:      {np.round(lin_reg.weights, 4)} (True: [2.5, -1.5])")
    print(f"Model Learned Bias:   {lin_reg.bias:.4f} (True: 3.0)")
    print(f"Mean Abs Error (MAE): {np.mean(np.abs(y_raw - y_pred_dec)):.6f}")

    # -------------------------------------------------------------
    # Test 4: Encrypted Logistic Regression (Classification)
    # -------------------------------------------------------------
    print(format_banner("4. Encrypted Logistic Regression Model Test"))
    X_cls = np.array([
        [-2.0, -1.5],
        [-1.0, -2.0],
        [1.5, 2.0],
        [2.0, 1.5]
    ])
    y_cls = np.array([0.0, 0.0, 1.0, 1.0])
    
    log_reg = EncryptedLogisticRegression(ctx, n_features=2)
    log_reg.fit(X_cls, y_cls, epochs=50, lr=0.2)
    
    X_cls_enc = EncryptedMatrix.encrypt(ctx, X_cls)
    probs_enc = log_reg.predict_proba(X_cls_enc)
    probs_dec = probs_enc.decrypt()[:len(y_cls)]
    
    print(f"True Classes Y:       {y_cls}")
    print(f"HE Predicted Probs:   {np.round(probs_dec, 4)}")
    preds_binary = (probs_dec >= 0.5).astype(float)
    print(f"HE Binary Decision:   {preds_binary}")
    print(f"Classification Acc:   {np.mean(preds_binary == y_cls) * 100:.1f}%")

    # -------------------------------------------------------------
    # Test 5: Encrypted Neural Network (Multi-Layer Perceptron)
    # -------------------------------------------------------------
    print(format_banner("5. Encrypted Multi-Layer Neural Network Test"))
    nn = EncryptedSequentialNetwork([
        EncryptedDenseLayer(in_features=2, out_features=3, activation='square', 
                            weights=[[0.5, -0.2, 0.8], [0.3, 0.6, -0.4]], bias=[0.1, -0.1, 0.2]),
        EncryptedDenseLayer(in_features=3, out_features=1, activation='poly_sigmoid',
                            weights=[[0.7], [-0.5], [0.9]], bias=[0.0])
    ])
    
    X_nn = np.array([[1.0, 0.5], [-0.5, 1.0], [0.8, -0.8]])
    X_nn_enc = EncryptedMatrix.encrypt(ctx, X_nn)
    
    nn_out_enc = nn.forward(X_nn_enc)
    nn_out_dec = nn_out_enc.decrypt()
    
    # Compute Plaintext reference
    # Layer 1
    z1 = X_nn @ np.array([[0.5, -0.2, 0.8], [0.3, 0.6, -0.4]]) + np.array([0.1, -0.1, 0.2])
    a1 = z1 ** 2
    # Layer 2
    z2 = a1 @ np.array([[0.7], [-0.5], [0.9]])
    a2 = 0.5 + 0.2328 * z2 - 0.0097 * (z2 ** 3)
    
    print(f"HE NN Predictions:    {np.round(nn_out_dec.flatten(), 4)}")
    print(f"Plaintext Ref Out:    {np.round(a2.flatten(), 4)}")
    print(f"NN Absolute Error:    {np.mean(np.abs(nn_out_dec.flatten() - a2.flatten())):.6f}")

    # -------------------------------------------------------------
    # Test 6: Encrypted K-Means Clustering Distance Calculation
    # -------------------------------------------------------------
    print(format_banner("6. Encrypted K-Means Distance Estimation Test"))
    X_km = np.array([
        [1.0, 1.0],
        [1.5, 1.8],
        [8.0, 8.0],
        [9.0, 8.5]
    ])
    
    kmeans = EncryptedKMeans(n_clusters=2)
    kmeans.fit(X_km)
    
    X_km_enc = EncryptedMatrix.encrypt(ctx, X_km)
    dists_enc = kmeans.compute_encrypted_distances(X_km_enc)
    dists_dec = dists_enc.decrypt()
    
    # Ground truth squared distance
    true_dists = np.array([[np.sum((x - c)**2) for c in kmeans.centroids] for x in X_km])
    
    print(f"Cluster Centroids:\n{np.round(kmeans.centroids, 2)}")
    print(f"HE Squared Distances:\n{np.round(dists_dec, 4)}")
    print(f"True Squared Distances:\n{np.round(true_dists, 4)}")
    print(f"Max Distance Error:   {np.max(np.abs(dists_dec - true_dists)):.6f}")

    print(format_banner("ALL HE-ML PACKAGE TESTS COMPLETED SUCCESSFULLY!"))

if __name__ == '__main__':
    run_all_tests()
