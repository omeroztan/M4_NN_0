import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential


class NeuralNetwork:
    def __init__(self, input_shape, param_list):
        # i wasn't sure whether to send a list of parameters, or send them piece by piece in variables

        # __init__(self,x_train, y_train, x_validation, y_validation):
        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_validation = x_validation
        # self.y_validation = y_validation

        #self.data_list = data_list
        #self.param_list = param_list

        self.LSTM_input_shape = input_shape  # (x_train, y_train, v_validation, y_validation)[0] = x_train
        self.model = self.build_nn

    def build_nn(self):
        print("\n")  # put end-line here, so it looks a bit more diffused, easier to read on console
        model = Sequential()
        model.add(LSTM(8, input_shape=(self.LSTM_input_shape[1:]), activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(4, activation='relu'))
        model.add(Dropout(0.1))

        model.add(Dense(1, activation='relu'))

        opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics=['mse'])
        return model


