from models.Logistic.CBOW_BOWLogistic import CBOW_BOW


class CBOW(CBOW_BOW):
    def __init__(self, language):
        super().__init__(language)

    def extract_X(self, dataset):
        return super().get_continuous_representation(dataset)