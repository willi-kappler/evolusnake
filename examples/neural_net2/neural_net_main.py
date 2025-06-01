# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging
import pathlib
from sys import float_info

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_select_population import es_select_population
from evolusnake.es_server import ESServer

from dataprovider import DataProvider, IterationNeural
from neural_net_1 import NeuralNetIndividual1
from neural_net_2 import NeuralNetIndividual2
from neural_net_3 import NeuralNetIndividual3
from neural_net_4 import NeuralNetIndividual4
from neural_net_5 import NeuralNetIndividual5

logger = logging.getLogger(__name__)


def load_data(filename: str) -> list:
    result = []

    min_val: float = float_info.max
    max_val: float = -float_info.max

    with open(filename, "r") as f:
        next(f)  # ignore column names
        for line in f:
            items: list = line.split(",")
            in1: float = float(items[1])
            in2: float = float(items[2])
            in3: float = float(items[3])
            in4: float = float(items[4])
            name: str = items[5].strip()
            kind = None

            match name:
                case "Iris-setosa":
                    kind = [1.0, 0.0, 0.0]
                case "Iris-versicolor":
                    kind = [0.0, 1.0, 0.0]
                case "Iris-virginica":
                    kind = [0.0, 0.0, 1.0]
                case _:
                    raise ValueError(f"Unknown name: '{name}'")

            result.append(([in1, in2, in3, in4], kind))

            min_val = min(min_val, in1, in2, in3, in4)
            max_val = max(max_val, in1, in2, in3, in4)

    max_val -= min_val

    # Normalize input values:
    for (data, _) in result:
        data[0] = (data[0] - min_val) / max_val
        data[1] = (data[1] - min_val) / max_val
        data[2] = (data[2] - min_val) / max_val
        data[3] = (data[3] - min_val) / max_val

    return result


def main():
    config: ESConfiguration = ESConfiguration.from_json("neural_net_config.json")
    config.from_command_line()

    server_mode = config.server_mode

    log_file_name: str = "neural_net_server.log"

    if not server_mode:
        neural_num: int = 1

        while True:
            log_file_name = f"neural_net_node_{neural_num}.log"
            p: pathlib.Path = pathlib.Path(log_file_name)

            if p.is_file():
                neural_num += 1
            else:
                break

    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("parasnake").setLevel(logging.WARNING)

    data_values = load_data("Iris.csv")

    dp = DataProvider(data_values, 20)

    net_kind: int = 1

    if config.user_options:
        opt = config.user_options
        # format for option: --user_options net=1
        (key, val) = opt.split("=")
        if key == "net":
            net_kind = int(val)
        else:
            raise ValueError(f"Unknown user option: {opt}")

    input_size: int = 4
    output_size: int = 3
    initial_size: int = 8  # -> Hyperparmeter
    # For classifyers alwasy use softmax!
    use_softmax: bool = True
    max_size: int = 12

    match net_kind:
        case 1:
            ind = NeuralNetIndividual1(input_size, output_size, dp, initial_size, use_softmax, max_size)
            config.mutation_operations = [0, 1, 2, 3, 4, 5, 6]
        case 2:
            ind = NeuralNetIndividual2(input_size, output_size, dp, initial_size, use_softmax, max_size)
            config.mutation_operations = [0, 1, 2, 3]
        case 3:
            ind = NeuralNetIndividual3(input_size, output_size, dp, initial_size, use_softmax, max_size)
            config.mutation_operations = [0, 1, 2, 3, 4, 5, 6, 7]
        case 4:
            ind = NeuralNetIndividual4(input_size, output_size, dp, initial_size, use_softmax, max_size)
            config.mutation_operations = [0, 1, 2, 3]
        case 5:
            ind = NeuralNetIndividual5(input_size, output_size, dp, initial_size, use_softmax, max_size)
            config.mutation_operations = [0, 1, 2, 3, 4, 5, 6]
        case _:
            raise ValueError(f"Unknown kind of neural net: {net_kind}")

    config.target_fitness = 0.0
    config.target_fitness2 = 0.01

    if server_mode:
        print("Create and start server.")
        server = ESServer(config, ind)
        server.ps_run()
    else:
        print("Create and start node.")
        logger.info(f"NeuralNet description: {ind.description()}")
        population = es_select_population(config, ind, IterationNeural())
        population.ps_run()


if __name__ == "__main__":
    main()
