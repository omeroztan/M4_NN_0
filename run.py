import data_prep
import neural_network
import nas
import test_data_inverse
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
list_of_MAPE = list()

size_daily = 4227
size_hourly = 414
size_monthly = 48000
size_quarterly = 24000
size_weekly = 359
size_yearly = 23000

for i in range(5):  # here size_daily etc. can be used but since it takes very long, I use a smaller number for testing

    # On my computer the paths are full, but on github this would not work. I add relative paths under
    # path_train = f'/Users/mac/Desktop/Datasets/M4_Dataset/M4_train_set/Daily_Train/daily_train_{i}.csv'
    # path_test = f'/Users/mac/Desktop/Datasets/M4_Dataset/M4_test_set/Daily_Test/daily_test_{i}.csv'

    path_train = f'daily_train_{i}.csv'
    path_test = f'daily_test_{i}.csv'

    timesteps = 3  # varied step size will be used

    # data operations
    training_dataframe = data_prep.DataPrep(csv_file_path=path_train, time_steps=timesteps, split_constant=0.9)
    testing_data = test_data_inverse.TestDataInverse(csv_file_path=path_test, timesteps=timesteps)

    x_train = training_dataframe.x_train
    y_train = training_dataframe.y_train
    x_validation = training_dataframe.x_validation
    y_validation = training_dataframe.y_validation
    data_list = [x_train, y_train, x_validation, y_validation]
    train = training_dataframe.train
    x_test, y_test = testing_data.x_test, testing_data.data

    # network architecture from NeuralNetwork class
    neural = neural_network.NeuralNetwork(input_shape=x_train.shape, param_list=2)  # only x_train.shape needed for now

    param = dict(epochs=[1000], batch_size=[16, 32, 64])

    # GridSearchCV object
    searcher = nas.Regres(model=neural.model, parameters=param)

    my_callbacks = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    searcher.grid.fit(x_train, y_train, validation_data=(x_validation, y_validation), callbacks=my_callbacks, verbose=0)
    print(searcher.grid.cv_results_['mean_test_score'],
          searcher.grid.cv_results_['std_test_score'],
          searcher.grid.cv_results_['params'])

    print(searcher.grid.score(x_test, y_test[timesteps:]))
    tr = MinMaxScaler()
    tr.fit(train)
    # in new sklearn, there is a function for calculating MAPE
    prediction = searcher.grid.predict(x_test)
    real_prediction_number = tr.inverse_transform(prediction.reshape(-1, 1))
    MAPE = mean_absolute_percentage_error(y_test[timesteps:], real_prediction_number)
    print("MAPE: ", MAPE)
    list_of_MAPE.append(MAPE)

print("Final MAPE List: ", list_of_MAPE)  # afterwards this list can be averaged over the dataset
