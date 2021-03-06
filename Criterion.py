class Criterion():
    def __init__(self, error, mode):
        self.error = error
        self.mode = mode

    def __call__(self, output, target):
        if self.mode == 'mse':
            return self.error.loss_mse(output, target)

        elif self.mode == 'SpikeTime':
            return self.error.spikeTime(output, target)
        elif self.mode == 'NumSpikes':
            return self.error.numSpikes(output, target)
        elif self.mode == 'ELBO':
            return self.error.ELBO(output, target)
        else:
            raise Exception("Mode not chosen correctly!")
