import numpy as np
from collections import defaultdict

from bglinking.general_utils import utils
from bglinking.graph.graph_rankers.InformalGraphRankerInterface import InformalGraphRankerInterface

# Modified from https://github.com/BrambleXu/news-graph

class DefaultGraphRanker(InformalGraphRankerInterface):
    def rank(self, nodes, edges) -> dict:   
        d = 0.85 # damping coefficient, usually is .85
        min_diff = 1e-5 # convergence threshold
        steps = 1000 # iteration steps
        weight_default = 1.0 / (len(nodes.keys()) or 1.0)

        nodeweight_dict = defaultdict(float) # store weight of node
        outsum_node_dict = defaultdict(float) # store weight of out nodes

        for node in nodes.values(): # initilize nodes weight by edges
            nodeweight_dict[node.name] = weight_default #node.weight 
             # Sum of all edges leaving a specific node
            outsum_node_dict[node.name] = sum((edges[edge_key] for edge_key in edges.keys() if node.name in edge_key))
        
        sorted_keys = sorted([node_name for node_name in nodes.keys()]) # save node name as a list for iteration
        step_dict = [0]
        for step in range(1, steps):
            new_weights = defaultdict(float)
            for edge, weight in edges.items():
                node_a, node_b = edge
                new_weights[node_a] += weight / outsum_node_dict[node_b] * nodeweight_dict[node_b]
                new_weights[node_b] += weight / outsum_node_dict[node_a] * nodeweight_dict[node_a]
            
            for node in sorted_keys:
                nodeweight_dict[node] = (1 - d) + d * new_weights[node]

            step_dict.append(sum(nodeweight_dict.values()))

            if abs(step_dict[step] - step_dict[step - 1]) <= min_diff:
                break

        # Normalize and Standardization
        if len(list(nodeweight_dict.values()))>0:
            nodeweight_dict = utils.standardize_dict(nodeweight_dict)
            nodeweight_dict = utils.normalize_dict(nodeweight_dict)

            assert np.min(list(nodeweight_dict.values()))>=0, nodeweight_dict
            
        for node in nodes.values():
            node.weight = nodeweight_dict[node.name]

        #return nodeweight_dict