from collections import defaultdict

from bglinking.graph.graph_builders.DefaultGraphBuilder import \
    DefaultGraphBuilder
from bglinking.graph.graph_comparators.GMCSComparator import GMCSComparator
from bglinking.graph.graph_rankers.DefaultGraphRanker import DefaultGraphRanker

# [modified] MIT license bramblu


class Graph:
    """Graph that represents a single news article."""

    def __init__(self, docid, fname):
        self.__nodes = {}  # {name: NodeObj1, name: NodeObj2, ...]
        self.__edges = defaultdict(float)  # {(A, B): weight}

        self.graph_builder = DefaultGraphBuilder()
        self.graph_ranker = DefaultGraphRanker()
        self.graph_comparator = GMCSComparator()

        self.docid = docid
        self.fname = fname

    def build(self, **kwargs):
        # Append local docid to kwargs.
        kwargs['docid'] = self.docid
        self.graph_builder.build(self, **kwargs)

    def rank(self):
        self.graph_ranker.rank(self.__nodes, self.__edges)

    def compare(self, graph, novelty_percentage, node_edge_l):
        return self.graph_comparator.compare(self, graph, novelty_percentage, node_edge_l)

    @property
    # @utils.limit_set
    def edges(self):
        return self.__edges

    @property
    # @utils.limit_set
    def nodes(self):
        return self.__nodes

    def add_node(self, node):
        """add node to nodes"""
        self.__nodes[node.name] = node

    def add_edge(self, start, end, weight):
        """Add edge between node"""
        # Store edge with nodes in alphabetical order.
        if start[0] < end[0]:
            self.__edges[(start, end)] = weight
        else:
            self.__edges[(end, start)] = weight

    def set_graph_builder(self, graph_builder):
        self.graph_builder = graph_builder

    def set_graph_ranker(self, graph_ranker):
        self.graph_ranker = graph_ranker

    def set_graph_comparator(self, graph_comparator):
        self.graph_comparator = graph_comparator

    def has_edge(self, edge) -> bool:
        return edge in self.edges.keys()

    def has_node(self, node) -> bool:
        return node in self.nodes.keys()

    def nr_nodes(self) -> int:
        return len(list(self.nodes.keys()))
