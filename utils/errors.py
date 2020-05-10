"""
Error handler
"""


class GradientMaskingError(ValueError):
    def __init__(self, arg):
        self.arg = arg
