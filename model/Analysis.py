from train import *
import os

class Analysis:
    def __init__(self):
        self.accuracies = []
        self.errors_train = []
        self.errors_valid = []
        self.test_accuracy = []
        self.results = {}

    def partial(self, neurons, rate, momentum, error_train, error_valid, accuracy):
        if neurons not in self.results:
            self.results[neurons] = {}
        if rate not in self.results[neurons]:
            self.results[neurons][rate] = {}
        self.results[neurons][rate][momentum] = {
            'error_train' : error_train,
            'error_valid' : error_valid,
            'accuracy' : accuracy
        }
        plt.plot(self.results[neurons][rate][momentum]['error_train'], label='Train Error')
        plt.plot(self.results[neurons][rate][momentum]['error_valid'], label='Valid Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title(f'Neurons: {neurons}, Learning Rate: {rate}, Momentum: {momentum}')
        plt.legend()
        plt.savefig(self.get_result_path_error(f"{neurons}-neurons-{rate}-rate-{momentum}-momentum"))
        plt.close()

    def get_result_path_error(self, name):
        return os.path.join(os.getcwd(), "results/errors", name + ".png")