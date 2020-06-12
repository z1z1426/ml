import numpy as np


class tree_node():
    def __init__(self, feat, val, right, left):
        self.feat = feat
        self.value = val
        self.right = right
        self.left = left
