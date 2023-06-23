from train import *
from model.Properties import *
from model.Dataset import *
from model.Analysis import *
import datetime

def main():
    properties = Properties("properties.ini")
    ds = Dataset(shuffle=True)
    analysis = Analysis()

    init_logs()
    write_logs(f"epochs for each training : {properties.epochs}")
    for neurons in properties.neurons:
        for rate in properties.learning_rate:
            for momentum in properties.momentum:
                layers = get_layers(neurons, momentum, ds.train_data.shape[0], properties.act_functions)
                results = train(ds, layers, rate, properties.epochs, properties.error_function)
                analysis.partial(neurons, rate, momentum, results[0], results[1], results[2])
                write_logs(f"end with: momentum {momentum}, learning_rate {rate}, neurons {neurons}, accuracy: {results[2].max()}")


def get_layers(neurons, momentum, columns, act_functions):
    return [Layer((neurons, columns), act_functions[0]['function'], act_functions[0]['derivative'], momentum), 
            Layer((10, neurons), act_functions[1]['function'], act_functions[1]['derivative'], momentum)
    ]

def init_logs():
    with open("results/events.log","a") as file:
        file.write(datetime.datetime.now().strftime('\n\nStarting test at %H:%M:%S - %d/%m/%Y'))

def write_logs(event):
    line = '\n' + event + datetime.datetime.now().strftime(', at %H:%M:%S - %d/%m/%Y')
    with open("results/events.log","a") as file:
        file.write(line)

if __name__ == "__main__":
    main()