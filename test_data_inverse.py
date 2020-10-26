from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class TestDataInverse:
    def __init__(self, csv_file_path, timesteps):
        self.data = pd.read_csv(csv_file_path, sep=',', index_col=0)
        self.time_steps = timesteps
        self.sc = MinMaxScaler(feature_range=(0.001, 1))

        self.normalized_data = self.sc.fit_transform(self.data)  # todo still not sure about what to write inside
        # self.inversed = self.sc.inverse_transform(self.__data)

        self.x_test, self.y_test = self.time_stepped()

    def time_stepped(self):
        '''Test sets does not need splitting, so they can be stepped directly'''
        steps = self.time_steps
        data = self.normalized_data
        x, y = list(), list()
        for i in range(len(self.normalized_data) - steps):
            seq_x = data[i:i+steps]
            seq_y = data[i+steps]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)
