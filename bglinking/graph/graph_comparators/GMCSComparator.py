import numpy as np
from scipy import stats
from collections import defaultdict
import operator
import warnings

from bglinking.general_utils import utils
from bglinking.graph import graph_utils
from bglinking.graph.graph_comparators.InformalGraphComparatorInterface import InformalGraphComparatorInterface


class GMCSComparator(InformalGraphComparatorInterface): 
    def type_distribution(self, nodes):
        distribution = defaultdict(float)
        for node in nodes.values():
            distribution[node.node_type] += 1
        return distribution

    def similarity(self, nodes_a, edges_a, nodes_b, edges_b, common_nodes, common_edges, node_edge_l) -> float:
        l = node_edge_l # node over edge importance
        sum_nodes_a = sum([node.weight for node in nodes_a.values()])
        sum_nodes_b = sum([node.weight for node in nodes_b.values()])

        sum_edges_a = sum([weight for weight in edges_a.values()])
        sum_edges_b = sum([weight for weight in edges_b.values()])
        
        nodes = l * (sum(common_nodes.values(), 0))/max(sum_nodes_a, sum_nodes_b, 1)
        edges = (1-l) * sum(common_edges.values(), 0)/max(sum_edges_a, sum_edges_b, 1)

        return nodes + edges


    def novelty(self, que_graph, can_graph, common_nodes) -> float:
        # determine weight thresholdhold! to be important node
        original_distribution = self.type_distribution(que_graph.nodes)
        new_info = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            weight_threshold = np.mean(list(can_graph.edges.values()))
        
        i = 0
        added_nodes = []
        for edge, weight in can_graph.edges.items():
            # overlap should be 1, meaning there is exactly 1 node in common nodes.
            if len(set(edge) & set(common_nodes)) == 1 and weight > weight_threshold:
                new_node = np.setdiff1d(edge, common_nodes)[0]
                if new_node not in added_nodes:
                    i += 1
                    new_info[new_node] = can_graph.nodes[new_node]
                    added_nodes.append(new_node)

        additional_distributions = self.type_distribution(new_info)

        not_matching_candidate_node_weights = [can_graph.nodes[key].weight for key in set(can_graph.nodes)-set(common_nodes)]
        if len(not_matching_candidate_node_weights) == 0:
            return (0.0, utils.get_keys_max_values(additional_distributions))

        additional_node_weights = [node.weight for node in new_info.values()]
        novelty = sum(additional_node_weights) / sum(not_matching_candidate_node_weights)

        return (novelty, utils.get_keys_max_values(additional_distributions))

    def compare(self, graph_a, graph_b, novelty_percentage, node_edge_l=0.5) -> (float, float):  
        """ Compare graph A with graph B and calculate similarity score.

        Returns
        -------
        (float, float)
            node similarity, edge similarity
        """        
        nodes_a = graph_a.nodes
        edges_a = graph_a.edges
        nodes_b = graph_b.nodes
        edges_b = graph_b.edges

        common_nodes = {node_name: node_obj.weight for node_name, node_obj in nodes_a.items() if node_name in nodes_b.keys()}
        common_edges = {edge: weight for edge, weight in edges_a.items() if edge in edges_b.keys()}

        similarity_score = (1 - novelty_percentage) * self.similarity(nodes_a, edges_a, nodes_b, edges_b, common_nodes, common_edges, node_edge_l)

        novelty_score, diversity_type = self.novelty(graph_b, graph_a, common_nodes)
        novelty_score = float(novelty_percentage * novelty_score)

        return stats.hmean([similarity_score+1e-05, novelty_score+1e-05]), diversity_type

    

