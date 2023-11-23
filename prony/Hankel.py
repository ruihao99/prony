import numpy as np

class Hankel:
    def __init__(self, time_domain_data: np.ndarray, nsample: int=2000):

        if self._is_valid_sample(time_domain_data,  nsample):
            raise ValueError()

    @staticmethod
    def _is_valid_sample(self, time_domain_data, nsample):
        ndata = len(time_domain_data)

        if nsample < ndata / 10:
            return False

        return True
