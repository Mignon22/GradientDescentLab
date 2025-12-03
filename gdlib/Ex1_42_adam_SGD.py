import numpy as np
import pandas as pd
from .base_experiment import BaseExperiment

# ======================================
#  2. Ex1-42 : Adam + SGD (一次一筆)
# ======================================
class Ex1_42_adam_SGD(BaseExperiment):
    def __init__(self, n_samples = 500, test_size = 0.2, master_seed = 42, lambda_reg = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lambda_reg = lambda_reg
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(n_samples = n_samples, test_size = test_size, master_seed = master_seed)
    
    def train_one_lr(self, initial_lr, iterations = 600, verbose = True, print_every = 50):
        self.initial_lr = initial_lr
        w = np.random.rand(self.X_train_standardized.shape[1])
        b = np.random.rand()

        lambda_reg = self.lambda_reg
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon

        m_w = np.zeros_like(w)
        m_b = 0
        v_w = np.zeros_like(w)
        v_b = 0

        lr_history = []
        loss_history_train = []
        loss_history_test = []
        w_history = []
        b_history = []

        for i in range(iterations):
            idx = np.random.randint(0, len(self.X_train_standardized))
            x_i = self.X_train_standardized[idx]
            y_i = self.y_train[idx]

            y_pred_i = np.dot(x_i , w) + b
            error_i = y_i - y_pred_i

            w_gradient = (-2) * x_i * error_i + 2 * lambda_reg * w
            b_gradient = (-2) * error_i

            m_w = beta1 * m_w + (1 - beta1) * w_gradient
            m_b = beta1 * m_b + (1 - beta1) * b_gradient
            v_w = beta2 * v_w + (1 - beta2) * (w_gradient ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (b_gradient ** 2)

            m_w_hat = m_w / (1 - beta1 ** (i+1))
            m_b_hat = m_b / (1 - beta1 ** (i+1))
            v_w_hat = v_w / (1 - beta2 ** (i+1))
            v_b_hat = v_b / (1 - beta2 ** (i+1))

            w -= initial_lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            b -= initial_lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            lr_history.append(initial_lr)
            w_history.append(w.copy())
            b_history.append(b)

            # === 計算 Train / Test Loss ===
            y_pred_train = np.dot(self.X_train_standardized, w) + b
            error_train = self.y_train - y_pred_train
            loss_train = np.mean(error_train ** 2) + lambda_reg * np.sum(w ** 2)
            loss_history_train.append(loss_train)

            y_pred_test = np.dot(self.X_test_standardized, w) + b
            error_test = self.y_test - y_pred_test
            loss_test = np.mean(error_test ** 2)
            loss_history_test.append(loss_test)

            if verbose and (i % print_every == 0 or i == (iterations-1)):
                print(f'Iteration {i} : w = {np.round(w, 4)}, b = {b:.4f}, Train Loss = {loss_train:.4f}, Test Loss = {loss_test:.4f}')
        
        # === 找出最小 Test Loss 對應的迭代點 ===
        loss_history_test_arr = np.array(loss_history_test)
        min_index = np.argmin(loss_history_test_arr)
        min_test_loss = loss_history_test_arr[min_index]
        best_w = w_history[min_index]
        best_b = b_history[min_index]
        lr_at_best = lr_history[min_index]

        df_summary = pd.DataFrame({
            'Final Train Loss' : [np.round(loss_history_train[iterations-1], 4)],
            'Final Test Loss' : [np.round(loss_history_test[iterations-1], 4)],
            'Best Iteration' : [min_index],
            'Train Loss @ Best Test' : [np.round(loss_history_train[min_index], 4)],
            'Best Test Loss' : [np.round(min_test_loss, 4)],
            'w (params)' : [np.round(best_w, 4)],
            'b (bias)' : [np.round(best_b, 4)],
            'Learning Rate @ Best Test' : [np.round(lr_at_best, 4)]
        })

        return min_test_loss, min_index, best_w, best_b, df_summary, loss_history_train, loss_history_test, lr_history
