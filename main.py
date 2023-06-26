from train import *
from model.Properties import *
from model.Dataset import *
from model.Analysis import *

def main():
    properties = Properties("properties.ini")
    ds = Dataset(shuffle=True)
    analysis = Analysis()

    analysis.init_logs()
    analysis.write_logs(f"epochs for each training : {properties.epochs}")
    for neurons in properties.neurons:
        for rate in properties.learning_rate:
            for momentum in properties.momentum:
                layers = get_layers(properties.hidden_layers, neurons, momentum, ds.train_data.shape[0], properties.act_functions)
                results = train(ds, layers, rate, properties.epochs, properties.error_function)
                analysis.partial(neurons, rate, momentum, results[0], results[1], results[2])
                analysis.write_logs(f"end with: momentum {momentum}, learning_rate {rate}, neurons {neurons}, accuracy: {results[2].max()}")


def get_layers(hidden_layers, neurons, momentum, columns, act_functions):
    layers = []
    row_layer, columns_layer = neurons, columns
    for i in range(hidden_layers):
        layers.append(Layer((row_layer, columns_layer), act_functions[i]['function'], act_functions[i]['derivative'], momentum))
        row_layer, columns_layer = columns_layer, row_layer

    layers.append(Layer((10, columns_layer), act_functions[-1]['function'], act_functions[-1]['derivative'], momentum))
    return layers

if __name__ == "__main__":
    main()