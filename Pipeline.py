from sklearn.model_selection import train_test_split


class Pipeline():
    def __init__(self, data):
        self.train_data, self.test_data = train_test_split(data, test_size=0.2)

    def train(self, model):
        """Train the model"""
        self.Y = self.train_data['is_answerable']
        self.model = model.fit(self.X, self.Y)
        print(f'Training accuracy: {self.model.score(self.X, self.Y)}')

    def validate(self):
        """Validate the model"""
        # self.X_validation = self.vectorizer.transform(self.validation_data['tokenized_question'])
        self.Y_validation = self.validation_data['is_answerable']
        print(f'Validation accuracy: {self.model.score(self.X_validation, self.Y_validation)}')
        

    
