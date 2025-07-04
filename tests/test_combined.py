# This file is part of Evolusnake, evolutionary algorithms in Python.
# written by Willi Kappler, MIT license.
#
# See: https://github.com/willi-kappler/evolusnake

# Python std lib:
import unittest

# Local imports:
from evolusnake.es_config import ESConfiguration
from evolusnake.es_individual import ESIndividual
from evolusnake.es_population_node1 import ESPopulationNode1
from evolusnake.es_population_node2 import ESPopulationNode2
# from evolusnake.es_population import ESPopulation

from evolusnake.es_server import ESServer
from tests.common import TestIndividual

# External imports:
from parasnake.ps_config import PSConfiguration
from parasnake.ps_nodeid import PSNodeId


class TestServer(unittest.TestCase):
    def test_server_and_node1(self):
        """
        Test server and node combined.
        """

        config1: ESConfiguration = ESConfiguration()
        config1.parasnake_config = PSConfiguration("12345678901234567890123456789012")
        config1.save_new_fitness = False
        config1.randomize_population = False
        config1.accept_new_best = True
        config1.share_only_best = True
        ind1: TestIndividual = TestIndividual()

        server1: ESServer = ESServer(config1, ind1)
        node_id1: PSNodeId = PSNodeId()

        population1: ESPopulationNode1 = ESPopulationNode1(config1, ind1)
        population2: ESPopulationNode2 = ESPopulationNode2(config1, ind1)

        ind_from_server: ESIndividual = server1.ps_get_new_data(node_id1)  # type: ignore
        result = population1.ps_process_data(ind_from_server)
        server1.ps_process_result(node_id1, result)

        ind_from_server: ESIndividual = server1.ps_get_new_data(node_id1)  # type: ignore
        result = population2.ps_process_data(ind_from_server)
        server1.ps_process_result(node_id1, result)
        job_done = server1.ps_is_job_done()

        self.assertTrue(job_done)


if __name__ == "__main__":
    unittest.main()

