import pandas as pd
import numpy as np

def lr_grid_search(experiment_class, lr_start = 0.2, lr_end = 8, lr_step = 0.01, iterations = 600, master_seed = 42, exp_kwargs = None):
    """
    簡潔通用版 LR grid search
    - experiment_class : Ex1_42_adam_SGD, Ex1_43_adam_SGDR, Ex1_44_adam_minibatch
    - exp_kwargs : 要傳給類別初始化的參數，例如 :
        {'n_samples' = 500, 'test_size' = 0.2, 'master_seed' = 42}
    """

    lr_list = np.arange(lr_start, lr_end + lr_step, lr_step)

    results = []
    best_lr = None
    best_loss = np.inf

    exp_kwargs = exp_kwargs or {}

    for lr in lr_list:
        print('Testing LR = ', np.round(lr, 2))
        
        # 重新建立實驗物件(確保資料集相同)
        exp = experiment_class(**exp_kwargs)

        # 呼叫 class 的 train_one_lr
        min_test_loss, *_ = exp.train_one_lr(initial_lr = lr, iterations = iterations, verbose = False)
        results.append({'lr' : lr, 'best_test_loss' : np.round(min_test_loss, 4)})

        if min_test_loss < best_loss:
            best_loss = min_test_loss
            best_lr = lr
    
    df = pd.DataFrame(results).sort_values('best_test_loss').reset_index(drop = True)
    
    return best_lr, df
