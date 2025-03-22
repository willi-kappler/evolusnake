# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

import logging

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population_node1 import ESPopulationNode1
from evolusnake.es_population_node2 import ESPopulationNode2
from evolusnake.es_population_node3 import ESPopulationNode3
from evolusnake.es_population_node4 import ESPopulationNode4
from evolusnake.es_population_node5 import ESPopulationNode5

# External imports:
from parasnake.ps_node import PSNode


logger = logging.getLogger(__name__)


def es_select_population(configuration: ESConfiguration, individual: ESIndividual) -> PSNode:
    pop_kind: int = configuration.population_kind

    match pop_kind:
        case 1:
            return ESPopulationNode1(configuration, individual)
        case 2:
            return ESPopulationNode2(configuration, individual)
        case 3:
            return ESPopulationNode3(configuration, individual)
        case 4:
            return ESPopulationNode4(configuration, individual)
        case 5:
            return ESPopulationNode5(configuration, individual)
        case _:
            raise ValueError(f"Unknown population kind: {pop_kind}")


