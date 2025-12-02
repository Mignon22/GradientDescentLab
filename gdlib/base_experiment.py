import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================
#  1. 基底類別 : 資料生成 & 標準化
# ======================================
class BaseExperiment:
    def __init__(self, n_samples = 500, test_size = 0.2, master_seed = 42):
        """
        BaseExperiment 負責 :
        - 產生合成房價資料
        - train / test 分割
        - 標準化 X
        - 把資料存成物件屬性，之後子類別直接用
        """
        self.n_samples = n_samples
        self.test_size = test_size
        self.master_seed = master_seed

        self._prepare_data()
    
    def _prepare_data(self):
        np.random.seed(self.master_seed)

        n_samples = self.n_samples
        area = np.random.uniform(10, 50, n_samples)
        expected_bedrooms = np.clip((area/15), 0, 4)
        bedrooms = np.random.normal(expected_bedrooms, 0.5)
        bedrooms = np.round(bedrooms).astype(int)
        bedrooms = np.clip(bedrooms, 0, 4)
        age = np.random.uniform(0, 30, n_samples)
        noise = np.random.normal(0, 2, n_samples)

        rent = 1.5 * area + 2 * bedrooms - 0.8 * age + 5 + noise

        X = np.column_stack((area, bedrooms, age))
        y = rent

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.master_seed)

        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train)
        X_test_standardized = scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_standardized = X_train_standardized
        self.X_test_standardized = X_test_standardized
        self.scaler = scaler
        self.mean = scaler.mean_
        self.scale = scaler.scale_

        print('=== Data Prepared ===')
        print('mean = ', np.round(self.mean, 4))
        print('scale = ', np.round(self.scale, 4))

