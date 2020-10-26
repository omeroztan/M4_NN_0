import neural_network
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping


class Regres:
    def __init__(self, model,  parameters):
        self.parameters = parameters
        self.model = model
        self.grid = self.do()


    def do(self):
        regressor = KerasRegressor(build_fn=self.model)
        grid = GridSearchCV(regressor, param_grid=self.parameters, cv=2)  # i need to solve this cv issue
        # my_callbacks = EarlyStopping(monitor='mse', patience=10, verbose=2)
        return grid
