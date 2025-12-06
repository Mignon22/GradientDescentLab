import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

def run_kfold(experiment_class, initial_lr, n_splits = 5, iterations = 25000, master_seed = 42, exp_kwargs = None, verbose = True):
    """
    通用 K-fold 驗證函式

    參數說明 :
    -------------------------------
    
    :param experiment_class: class
        例如 Ex1_42_adam_SGD / Ex1_43_adam_SGDR / Ex1_44_adam_MB
        類別裡面需要實作 train_one_lr(...), 回傳:
        (min_test_loss, min_index, best_w, best_b, df_summary, loss_history_train, loss_history_test, lr_history)
    :param initial_lr: float
        每個 fold 訓練時用的 initial_lr
    :param k_fold: int
        K-fold 中的 K, 預設 5
    :param iterations: int
        每個 fold 訓練迭代次數
    :param master_seed: int
        控制 K-fold 分割用的亂數種子 (確保可重現)
    :param exp_kwargs: dict 或 None
        建立 experiment_class 時要傳入的參數，例如:
        {
            'n_samples': 500,
            'test_size': 0.2,
            'master_seed': 42,
            'lambda_reg': 0.001,
        }
    :param verbose: bool
        是否列印訓練過程

    回傳:
    -------------------------------
    df_folds : DataFrame
        每個 fold 的摘要 (fold, best_test_loss, best_iteration, final_train_loss, final_test_loss ...)

    mean_best_loss : float
        各 fold Best Test Loss 的平均
    
    std_best_loss : float
        各 fold Best Test Loss 的標準差
    """

    exp_kwargs = exp_kwargs or {}

    # 先取出原始訓練資料 (排除測試集)
    # 建立一次實驗物件，只用它生成資料，不做訓練
    tmp_exp = experiment_class(**exp_kwargs)
    X_full = tmp_exp.X_train_standardized
    y_full = tmp_exp.y_train

    kf = KFold(n_splits = n_splits, shuffle = True, random_state = master_seed)

    fold_results = []
    fold_id = 1

    for train_idx, val_idx in kf.split(X_full):
        print(f'\n==== Fold {fold_id} / {n_splits} ====')

        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]

        X_val_fold = X_full[val_idx]
        y_val_fold = y_full[val_idx]

        # 重新建立實驗物件 (資料會一致，但權重初始化會重新進行)
        exp = experiment_class(**exp_kwargs)

        # 覆寫資料，避免 BaseExperiment 重新劃分, 重新劃分 train / test
        exp.X_train_standardized = X_train_fold
        exp.y_train = y_train_fold
        exp.X_test_standardized = X_val_fold
        exp.y_test = y_val_fold

        # 執行訓練
        min_test_loss, min_index, best_w, best_b, df_summary, loss_history_train, loss_history_test, lr_history = exp.train_one_lr(initial_lr = initial_lr, iterations = iterations, verbose = verbose)

        # 儲存結果
        fold_results.append({
            'fold' : fold_id,
            'best_test_loss' : np.round(min_test_loss, 4),
            'best iteration' : min_index,
            'Final_train_loss' : np.round(df_summary['Final Train Loss'][0], 4),
            'Final_test_loss' : np.round(df_summary['Final Test Loss'][0], 4),
            'best_w' : np.round(best_w, 4),
            'best_b' : np.round(best_b, 4)
        })

        print(f'Fold {fold_id} -- Best Test Loss = {min_test_loss:.4f}')
        fold_id += 1
    
    # 建立 summary DataFrame
    df_folds = pd.DataFrame(fold_results)

    mean_best_loss = df_folds['best_test_loss'].mean()
    std_best_loss = df_folds['best_test_loss'].std()

    print('\n======= K-fold =======')
    print(df_folds)
    print(f'\nMean Best Test Loss = {mean_best_loss:.4f}')
    print(f'Std Best Test Loss = {std_best_loss:.4f}')

    return df_folds, mean_best_loss, std_best_loss



def kfold_search_lr(experiment_class, lr_list, n_splits = 5, iterations = 25000, exp_kwargs = None, master_seed = 42, verbose = True):
    """
    搜尋最佳 learning rate 的 K-fold 驗證流程:
    
    參數:
    ----------------------------
    :param experiment_class: class
        Ex1_42_adam_SGD / Ex1_43_adam_SGDR / Ex1_44_adam_MB
    :param lr_list: list
        要測試的學習率，例如: [0.01, 0.02, 0.04, 0.08]
    :param n_splits: int
        幾折驗證 (預設 5)
    :param iterations: int
        每 fold 訓練次數
    :param exp_kwargs: dict or None
        初始化 experiment_class 用的其他參數
    :param master_seed: int
        控制 K-fold 分割用的亂數種子 (確保可重現)
    :param verbose: bool
        是否列印訓練過程
    """

    exp_kwargs = exp_kwargs or {}
    lr_results = []

    print('\n===== K-fold Learning Rate Search =====')

    for lr in lr_list:
        print(f'\n Testing LR = {lr}')

        df_folds, mean_best_loss, std_best_loss = run_kfold(experiment_class = experiment_class, initial_lr = lr, n_splits = n_splits, iterations = iterations, master_seed = master_seed, exp_kwargs = exp_kwargs, verbose = verbose)
        
        lr_results.append({
            'lr': lr,
            'mean_best_loss': np.round(mean_best_loss, 4),
            'std_best_loss': np.round(std_best_loss, 4)
        })
    
    df_lr = pd.DataFrame(lr_results).sort_values('mean_best_loss').reset_index(drop = True)
    best_lr = df_lr.iloc[0]['lr']

    print('\n===== LR Search Results =====')
    print(df_lr)
    print(f'\nBest LR = {best_lr:.2f}')

    return best_lr, df_lr



