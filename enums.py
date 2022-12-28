class EnumBase:
    @classmethod
    def get_list(cls):
        return [getattr(cls, attr) for attr in dir(cls) if attr.isupper()]

    @classmethod
    def get_dict(cls):
        return {attr.lower(): getattr(cls, attr) for attr in dir(cls) if attr.isupper()}


class Path(EnumBase):
    DATA_ROOT = "./data"
    MNIST = "/mnist.csv"


class Setting(EnumBase):
    NUM_USERS = 10
    NUM_CHOSEN_USERS = 10

    NUM_EPOCHS = 25

    MODEL = "mlp"
    DATASET = "mnist"

    CLIPPING_THRESHOLD = 1.8

    IID = True


class MNIST(EnumBase):
    NUM_CLASSES = 10


class DP(EnumBase):
    PRIVACY_BUDGET = []  # epsilon
