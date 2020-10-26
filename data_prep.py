import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataPrep:
    """Data Preparation Class, I want to use an instance of this class to get the data network needs"""
    def __init__(self, csv_file_path, time_steps, split_constant):
        self.csv_file_path = csv_file_path
        self.time_steps = time_steps
        self.split_constant = split_constant

        self.__train, self.__validation = self.split()  # a bit confusion woth train, validation. I do not call them??
        self.__scaled_train, self.__scaled_validation = self.normalize()
        self.__data = self.data()

        self.x_train, self.y_train = self.time_steps_train()
        self.x_validation, self.y_validation = self.time_steps_validation()

        # sc = MinMaxScaler()
        # self.normalizer_whole_file = sc.fit(self.__data)
        # self.inversed = sc.inverse_transform(self.__data)

    def data(self):
        """Getting the data as a DataFrame from file path"""
        path = self.csv_file_path
        data = pd.read_csv(path, sep=',', index_col=0)
        data = data.values
        return data

    def normalize(self):
        """Normalising the data after splitting them, so there is no information leak from future set"""
        train, validation = self.split()
        scaler = MinMaxScaler(feature_range=(0.001, 1))
        scaled_train = scaler.fit_transform(train)  # todo ask about this normalization step, using train on both sets
        scaled_validation = scaler.fit_transform(train)
        return scaled_train, scaled_validation

    def split(self):
        """Splitting data to train and validation sets"""
        split_const = self.split_constant
        cut = int(split_const * len(self.data()))
        train = self.data()[:cut]
        validation = self.data()[-cut:]
        return train, validation

    def time_steps_train(self):
        """Preparing the data to have a time_step fashion"""
        steps = self.time_steps
        data = self.__scaled_train
        x, y = list(), list()
        for i in range(len(self.__scaled_train) - steps):
            seq_x = data[i:i+steps]
            seq_y = data[i+steps]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    def time_steps_validation(self):
        """Preparing Validation data"""
        steps = self.time_steps
        data = self.__scaled_validation
        x, y = list(), list()
        for i in range(len(self.__scaled_validation) - steps):
            seq_x = data[i:i+steps]
            seq_y = data[i+steps]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)




