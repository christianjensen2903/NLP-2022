from models.Model import Model
from xgboost import XGBClassifier as XGB_model
from pickle import dump, load

class XGBoost(Model):
    def __init__(self):
        super().__init__()
        self.model = XGB_model(objective='binary:logistic', random_state=43) #  tree_method='gpu_hist', 

    def train(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def save(self):
        dump(self.model, open(self.get_save_path('pkl'), 'wb'))

    def load(self):
        self.model = load(open(self.get_save_path('pkl'), 'rb'))