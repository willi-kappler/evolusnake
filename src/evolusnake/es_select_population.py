# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population import ESIterationCallBack
from evolusnake.es_population_node1 import ESPopulationNode1
from evolusnake.es_population_node2 import ESPopulationNode2
from evolusnake.es_population_node3 import ESPopulationNode3
from evolusnake.es_population_node4 import ESPopulationNode4
from evolusnake.es_population_node5 import ESPopulationNode5
from evolusnake.es_population_node6 import ESPopulationNode6
from evolusnake.es_population_node7 import ESPopulationNode7
from evolusnake.es_population_node8 import ESPopulationNode8
from evolusnake.es_population_node9 import ESPopulationNode9
from evolusnake.es_population_node10 import ESPopulationNode10
from evolusnake.es_population_node11 import ESPopulationNode11

# External imports:
from parasnake.ps_node import PSNode


logger = logging.getLogger(__name__)


def es_select_population(configuration: ESConfiguration, individual: ESIndividual,
        iteration_callback: ESIterationCallBack = ESIterationCallBack()) -> PSNode:
    pop_kind: int = configuration.population_kind

    match pop_kind:
        case 1:
            return ESPopulationNode1(configuration, individual, iteration_callback)
        case 2:
            return ESPopulationNode2(configuration, individual, iteration_callback)
        case 3:
            return ESPopulationNode3(configuration, individual, iteration_callback)
        case 4:
            return ESPopulationNode4(configuration, individual, iteration_callback)
        case 5:
            return ESPopulationNode5(configuration, individual, iteration_callback)
        case 6:
            return ESPopulationNode6(configuration, individual, iteration_callback)
        case 7:
            return ESPopulationNode7(configuration, individual, iteration_callback)
        case 8:
            return ESPopulationNode8(configuration, individual, iteration_callback)
        case 9:
            return ESPopulationNode9(configuration, individual, iteration_callback)
        case 10:
            return ESPopulationNode10(configuration, individual, iteration_callback)
        case 11:
            return ESPopulationNode11(configuration, individual, iteration_callback)
        case _:
            raise ValueError(f"Unknown population kind: {pop_kind}")


