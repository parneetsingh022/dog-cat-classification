from transformers import PretrainedConfig

class CustomModelConfig(PretrainedConfig):
    model_type = "custom"
    input_size = (3, 128, 128)
    num_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = kwargs.get("input_size", self.input_size)
        self.num_classes = kwargs.get("num_classes", self.num_classes)
