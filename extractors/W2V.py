from extractors.Extractor import Extractor


class W2V(Extractor):
    def __init__(self, language, dataset):
        super().__init__(language, dataset)

    def run(self, data):
        X = (
            data['tokenized_question'].to_list() +
            data['tokenized_plaintext'].to_list()
        )
        y = None
        return X, y
