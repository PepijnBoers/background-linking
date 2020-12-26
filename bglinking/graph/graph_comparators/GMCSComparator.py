import warnings
from collections import defaultdict

import numpy as np
from bglinking.general_utils import utils
from bglinking.graph.graph_comparators.InformalGraphComparatorInterface import (
    InformalGraphComparatorInterface,
)
from scipy import stats


class GMCSComparator(InformalGraphComparatorInterface):
    def similarity(self, graph_a, graph_b, common_nodes, common_edges, node_edge_l):
        """Return similarity score between two graphs based on the Greatest Common Subgraph.

        Parameters
        ----------
        graph_a : Graph
            Graph A
        graph_b : Graph
            Graph B
        common_nodes : dict
            Overlapping nodes graph A & B
        common_edges : dict
            Overlapping edges graph A & B
        node_edge_l : float
            Node-edge priority; higher is node priority

        Returns
        -------
        float
            Similarity score
        """
        sum_nodes_a = sum([node.weight for node in graph_a.nodes.values()])
        sum_nodes_b = sum([node.weight for node in graph_b.nodes.values()])
        sum_edges_a = sum([weight for weight in graph_a.edges.values()])
        sum_edges_b = sum([weight for weight in graph_b.edges.values()])

        # Similarity nodes.
        nodes = (
            node_edge_l
            * (sum(common_nodes.values(), 0))
            / max(sum_nodes_a, sum_nodes_b, 1)
        )

        # Similarity edges.
        edges = (
            (1 - node_edge_l)
            * sum(common_edges.values(), 0)
            / max(sum_edges_a, sum_edges_b, 1)
        )

        return nodes + edges

    def novelty(self, can_graph, common_nodes) -> float:
        # Determine weight threshold for novel nodes (mean).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            weight_threshold = np.mean(list(can_graph.edges.values()))

        # Extract novel nodes.
        novel_nodes = extract_novel_nodes(can_graph, common_nodes, weight_threshold)

        # Calculate novelty & total node score
        novelty_score = sum(novel_nodes.values())
        total_graph_score = sum([node.weight for node in can_graph.nodes.values()])

        # Normalize score
        normalized_novelty_score = novelty_score / (total_graph_score + 1e-5)

        return normalized_novelty_score

    def compare(self, graph_a, graph_b, novelty_percentage, node_edge_l=0.5) -> float:
        """Calculate relevance score for graph A with graph B based on:
        - similarity score
        - novelty score

        Returns
        -------
        (float, float)
            Relevance score, diversity type
        """
        # Create dictionary of common nodes {node_name: node_weight}.
        common_nodes = {
            node_name: node_obj.weight
            for node_name, node_obj in graph_a.nodes.items()
            if node_name in graph_b.nodes.keys()
        }

        # Create dictionary of common edges {edge: weight}.
        common_edges = {
            edge: weight
            for edge, weight in graph_a.edges.items()
            if edge in graph_b.edges.keys()
        }

        # Calculate similarity score between common nodes and edges.
        similarity_score = (1 - novelty_percentage) * self.similarity(
            graph_a, graph_b, common_nodes, common_edges, node_edge_l
        )

        # Calculate novelty score between two graphs.
        novelty_score = self.novelty(graph_a, common_nodes)
        novelty_score = float(novelty_percentage * novelty_score)

        # Calculate harmonic mean between similarity and nvoelty.
        harmonic_mean = stats.hmean([similarity_score + 1e-05, novelty_score + 1e-05])

        return harmonic_mean


def extract_novel_nodes(graph, common_nodes, weight_threshold):
    novel_nodes = {}
    # Search for novel nodes.
    for edge, weight in graph.edges.items():
        # overlap should be 1, meaning there is only 1 connection.
        if len(set(edge) & set(common_nodes)) == 1 and weight > weight_threshold:
            novel_node = np.setdiff1d(edge, common_nodes)[0]
            if novel_node not in novel_nodes.keys():
                novel_nodes[novel_node] = graph.nodes[novel_node].weight
    return novel_nodes  # {"term": weight, ..}
