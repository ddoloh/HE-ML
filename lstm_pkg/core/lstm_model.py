import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def dsigmoid(y):
    return y * (1.0 - y)

def dtanh(y):
    return 1.0 - y * y

class LSTMSequencePredictor:
    """
    LSTM Deep Learning Model for Time-Series Sequence Prediction and Forecasting.
    Implements full LSTM cell architecture with BPTT / Gradient Descent training.
    """
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, seed=42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        concat_dim = hidden_dim + input_dim
        
        scale = 1.0 / np.sqrt(hidden_dim)
        # Weights and biases for gates
        self.Wf = np.random.uniform(-scale, scale, (hidden_dim, concat_dim))
        self.bf = np.ones((hidden_dim, 1)) # Forget gate bias initialized to 1.0
        
        self.Wi = np.random.uniform(-scale, scale, (hidden_dim, concat_dim))
        self.bi = np.zeros((hidden_dim, 1))
        
        self.Wc = np.random.uniform(-scale, scale, (hidden_dim, concat_dim))
        self.bc = np.zeros((hidden_dim, 1))
        
        self.Wo = np.random.uniform(-scale, scale, (hidden_dim, concat_dim))
        self.bo = np.zeros((hidden_dim, 1))
        
        self.Wy = np.random.uniform(-scale, scale, (output_dim, hidden_dim))
        self.by = np.zeros((output_dim, 1))
        
        self.scaler_min = 0.0
        self.scaler_max = 1.0

    def forward_sequence(self, x_seq):
        """
        Forward pass over a single sequence x_seq of shape (seq_len, input_dim).
        Returns predicted y_pred, states cache.
        """
        seq_len = len(x_seq)
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        
        caches = []
        for t in range(seq_len):
            xt = x_seq[t].reshape(-1, 1)
            z = np.vstack((h, xt))
            
            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            c_tilde = np.tanh(self.Wc @ z + self.bc)
            
            c_next = f * c + i * c_tilde
            o = sigmoid(self.Wo @ z + self.bo)
            h_next = o * np.tanh(c_next)
            
            caches.append((xt, z, f, i, c_tilde, c, c_next, o, h_next))
            h = h_next
            c = c_next
            
        y_pred = self.Wy @ h + self.by
        return y_pred, caches

    def fit(self, X, y, epochs=30, lr=0.01, verbose=False):
        """
        Trains the LSTM model on sequence windows X shape (N, seq_len, 1) and y shape (N, 1).
        Tracks training loss history per epoch.
        """
        # Normalize target & features
        self.scaler_min = float(np.min(X))
        self.scaler_max = float(np.max(X)) + 1e-8
        
        X_norm = (X - self.scaler_min) / (self.scaler_max - self.scaler_min)
        y_norm = (y - self.scaler_min) / (self.scaler_max - self.scaler_min)
        
        loss_history = []
        n_samples = len(X_norm)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Gradient accumulators
            dWf = np.zeros_like(self.Wf)
            dbf = np.zeros_like(self.bf)
            dWi = np.zeros_like(self.Wi)
            dbi = np.zeros_like(self.bi)
            dWc = np.zeros_like(self.Wc)
            dbc = np.zeros_like(self.bc)
            dWo = np.zeros_like(self.Wo)
            dbo = np.zeros_like(self.bo)
            dWy = np.zeros_like(self.Wy)
            dby = np.zeros_like(self.by)
            
            for idx in range(n_samples):
                x_seq = X_norm[idx]
                target = y_norm[idx].reshape(-1, 1)
                
                y_pred, caches = self.forward_sequence(x_seq)
                err = y_pred - target
                loss = float(0.5 * np.sum(err ** 2))
                total_loss += loss
                
                # Output layer gradient
                last_h = caches[-1][8]
                dWy += err @ last_h.T
                dby += err
                
                dh_next = self.Wy.T @ err
                dc_next = np.zeros((self.hidden_dim, 1))
                
                # Backpropagation through time
                for t in reversed(range(len(x_seq))):
                    xt, z, f, i, c_tilde, c_prev, c_curr, o, h_curr = caches[t]
                    
                    dh = dh_next
                    tanh_c = np.tanh(c_curr)
                    
                    do = dh * tanh_c
                    do_input = do * dsigmoid(o)
                    dWo += do_input @ z.T
                    dbo += do_input
                    
                    dc = dc_next + (dh * o * dtanh(tanh_c))
                    
                    df = dc * c_prev
                    df_input = df * dsigmoid(f)
                    dWf += df_input @ z.T
                    dbf += df_input
                    
                    di = dc * c_tilde
                    di_input = di * dsigmoid(i)
                    dWi += di_input @ z.T
                    dbi += di_input
                    
                    dc_tilde = dc * i
                    dc_tilde_input = dc_tilde * dtanh(c_tilde)
                    dWc += dc_tilde_input @ z.T
                    dbc += dc_tilde_input
                    
                    dz = (self.Wf.T @ df_input) + (self.Wi.T @ di_input) + (self.Wc.T @ dc_tilde_input) + (self.Wo.T @ do_input)
                    dh_next = dz[:self.hidden_dim]
                    dc_next = dc * f
                    
            # Update weights
            self.Wf -= lr * (dWf / n_samples)
            self.bf -= lr * (dbf / n_samples)
            self.Wi -= lr * (dWi / n_samples)
            self.bi -= lr * (dbi / n_samples)
            self.Wc -= lr * (dWc / n_samples)
            self.bc -= lr * (dbc / n_samples)
            self.Wo -= lr * (dWo / n_samples)
            self.bo -= lr * (dbo / n_samples)
            self.Wy -= lr * (dWy / n_samples)
            self.by -= lr * (dby / n_samples)
            
            avg_loss = total_loss / n_samples
            loss_history.append(avg_loss)
            
        return loss_history

    def predict(self, X):
        """Predicts output values for input sequence batch X."""
        X_norm = (X - self.scaler_min) / (self.scaler_max - self.scaler_min)
        preds = []
        for x_seq in X_norm:
            y_pred, _ = self.forward_sequence(x_seq)
            preds.append(y_pred[0, 0])
            
        preds_norm = np.array(preds).reshape(-1, 1)
        return preds_norm * (self.scaler_max - self.scaler_min) + self.scaler_min

    def forecast_sampling(self, initial_seq, n_steps=20):
        """
        Autoregressively forecasts future time-series values given an initial sequence window.
        """
        curr_seq = np.array(initial_seq, dtype=float).copy().reshape(-1, 1)
        forecasts = []
        
        for _ in range(n_steps):
            window = curr_seq[-len(initial_seq):]
            X_win = window.reshape(1, len(initial_seq), 1)
            pred = self.predict(X_win)[0, 0]
            forecasts.append(pred)
            curr_seq = np.vstack((curr_seq, [[pred]]))
            
        return np.array(forecasts)
