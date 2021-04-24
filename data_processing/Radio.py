import abc


class Radio:
    """ Main transceiver class """

    def __init__(self,
                 freq,
                 bw_signal,
                 init_gain,
                 max_gain,
                 **kwargs):

        # save positional arguments
        self.freq = freq
        self.bw_signal = bw_signal
        assert init_gain <= max_gain, "init_gain > max_gain"
        self.init_gain = init_gain
        self.gain = init_gain
        self.max_gain = max_gain

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, type, value, traceback):
        pass

    def update(self, **kwargs):
        """Saves keyword arguments as member variables"""
        for param, value in kwargs.items():
            if(param not in ['gain', 'freq']):  # ignore frequency and gain (tuned)
                self.__setattr__(param, value)

    def get_attributes(self):
        """returns a dictionary with all attributes for logging purposes"""
        return self.__dict__.copy()
