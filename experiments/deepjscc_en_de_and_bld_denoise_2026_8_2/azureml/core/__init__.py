"""No-op subset of azureml.core required by BinaryLatentDiffusion logging."""


class Run:
    @staticmethod
    def get_context():
        return None
