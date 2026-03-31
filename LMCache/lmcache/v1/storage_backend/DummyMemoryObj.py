import torch
class DummyMemoryObj:

    def __init__(self,tensor):
        self._tensor = tensor

    def get_tensor(self, index=0):
        return self._tensor

    def get_num_tokens(self):
        return self._tensor.shape[-2]

    def is_valid(self):
        return self._tensor is not None
