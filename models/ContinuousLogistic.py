from models.ContinuousBOWLogistic import ContinuousBOWLogistic


class ContinuousLogistic(ContinuousBOWLogistic):
    def __init__(self):
        super().__init__()

    def extract_X(self, dataset, language):
        return super().get_continuous_representation(
            dataset,
            language
        )