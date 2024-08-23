import pytest
import torch

from resnet50_cefas.model import load_model

@pytest.fixture
def single_input():
    return torch.rand([1, 3, 50, 45], dtype=torch.float32)

@pytest.fixture
def loaded_model():
    model = load_model()
    return model

@pytest.fixture
def loaded_model_stripped():
    model = load_model(strip_final_layer=True)
    return model

def test_forward(loaded_model, single_input):
    output = loaded_model(single_input)
    assert output.shape == (1, 3)

def test_forward_stripped(loaded_model_stripped, single_input):
    output = loaded_model_stripped(single_input)
    assert output.shape == (1, 2048)

