from models.Model import Model

class SequenceLabeller(Model):

    def _tag_token(self, token, index, answer_start, answer_end):
            """Tag a token with the IOB format"""
            if index == answer_start:
                return (token, 'B')
            elif answer_start < index <= answer_end:
                return (token, 'I')
            else:
                return (token, 'O')

    def _tag_sentence(self, sentence, answer_start, answer_end):
        """Tag a sentence with the IOB format"""
        return [
            self._tag_token(token, index, answer_start, answer_end)
            for index, token in enumerate(sentence)
        ]

    def convert_to_iob(self, dataset):
        """Tag the dataset with the IOB format"""

        dataset['tagged_plaintext'] = dataset.apply(lambda row: self._tag_sentence(row['tokenized_plaintext'], row['answer_start'], row['answer_end']), axis=1)

        return dataset

    def extract_X(self, dataset, language: str = ""):
        """Extract features from the dataset"""
        pass

    def train(self, X, y):
        """Train the model"""
        pass

    def predict(self, X):
        """Predict the answer"""
        pass

    def evaluate(self, X, y):
        """Evaluate the model"""
        pass

    def save(self, language: str):
        """Save the model"""
        pass

    def load(self, language: str):
        """Load the model"""
        pass
