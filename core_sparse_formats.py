import numpy as np
import pandas as pd

class SparseMatrix:
    def __init__(self, shape, dtype = float):
        self.shape = shape
        self.data = []