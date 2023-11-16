
# 这是一个自动生成的Python文件
def hello_world():
    print("Hello, world! Time is 'Fri Nov 10 15:08:56 2023'")


if __name__ == "__main__":
    hello_world()
    b = 1
            
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper


