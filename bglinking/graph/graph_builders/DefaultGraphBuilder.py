import json
import logging

import numpy as np
from bglinking.database_utils import db_utils as db_utils
from bglinking.general_utils import utils
from bglinking.graph.graph_builders.InformalGraphBuilderInterface import (
    InformalGraphBuilderInterface,
)
from bglinking.graph.Node import Node


class DefaultGraphBuilder(InformalGraphBuilderInterface):
    def build(
        self,
        graph,
        cursor,
        embeddings,
        index_utils,
        docid,
        use_entities,
        nr_terms=0,
        term_tfidf=0.0,
        term_position=0.0,
        text_distance=0.0,
        term_embedding=0.0,
    ):
        # Retrieve named entities from database.
        if use_entities:
            entities = db_utils.get_entities_from_docid(cursor, docid, "entity_ids")

            # Create nodes for named entities
            # [['Washington Redskins', '[30]', '1', 'ORG']]
            for entity in entities:
                ent_name = entity[0]
                try:
                    ent_positions = json.loads(entity[1])
                except Exception as e:
                    print(f"issue with enitity: {entity[1]}")
                    logging.exception(e)
                    continue
                ent_tf = int(entity[2])
                ent_type = entity[3]

                graph.add_node(Node(ent_name, ent_type, ent_positions, ent_tf))

        # Retrieve top n tfidf terms from database.
        if nr_terms > 0.0:
            terms = db_utils.get_entities_from_docid(cursor, docid, "tfidf_terms")[
                :nr_terms
            ]

            # Create nodes for tfidf terms
            for term in terms:  # [['Washington Redskins', '[30]', '1', 'ORG']]
                term_name = term[0]
                term_positions = json.loads(term[1])
                term_tf = int(term[2])
                graph.add_node(Node(term_name, "term", term_positions, term_tf))

        # Determine node weighs
        calculate_node_weights(graph, docid, index_utils, term_tfidf, term_position)

        # Enity weights differ from terms in magnitide,
        # since they are tf only (normalize both individually).
        equalize_term_and_entities(graph)

        # Initialize edges + weights
        initialize_edges(graph, text_distance, term_embedding, embeddings)


def tf_func(node, N: int) -> float:
    if node.tf == 1:
        return 1 / N
    else:
        return (1 + np.log(node.tf - 1)) / N


def position_in_text(node: Node, docid: str, index_utils) -> float:
    # location refers to the earliest paragraph index (starts at 0, so +1).
    return 1 / (min(node.locations) + 1)


def distance_in_text(distance: int) -> float:
    if distance <= 1:
        return 1 / (1.0 + distance)
    else:
        return 0.0


def edge_embedding_weight(node_a, node_b, embeddings, embeddings_not_found):
    try:
        similarity = embeddings.similarity(node_a.__str__(), node_b.__str__())
    except Exception as e:
        # print(f'Wd did not found: {node_a.__str__()}, {node_b.__str__()}')
        similarity = 0
        logging.exception(e)
    return similarity


def closest_distance(node_a, node_b):
    min_distance = 999999
    for locations in node_a.locations:
        for other_locations in node_b.locations:
            distance = abs(locations - other_locations)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def equalize_term_and_entities(graph):
    """Normalize NE weights and tf-idf weights to obtain comparable weight."""
    term_weights = {"dummy0000": 0.0}
    entity_weights = {"dummy0000": 0.0}

    for node in graph.nodes.values():
        if node.node_type == "term":
            term_weights[node] = node.weight
        else:
            entity_weights[node] = node.weight

    normalized_term_weights = utils.normalize_dict(term_weights)
    normalized_entity_weights = utils.normalize_dict(entity_weights)

    del normalized_term_weights["dummy0000"]
    del normalized_entity_weights["dummy0000"]

    for node in graph.nodes.values():
        if node.node_type == "term":
            node.weight = normalized_term_weights[node]
        else:
            node.weight = normalized_entity_weights[node]


def calculate_node_weights(graph, docid, index_utils, term_tfidf, term_position):
    N = graph.nr_nodes()
    n_stat = index_utils.stats()["documents"]
    for node_name, node in graph.nodes.items():
        weight = 0.0
        if term_tfidf > 0:
            tf = tf_func(node, N)

            if node.node_type == "term":
                df = (
                    index_utils.get_term_counts(
                        utils.clean_NE_term(node_name), analyzer=None
                    )[0]
                    + 1e-5
                )
                weight += utils.tfidf(tf, df, n_stat)
            else:
                weight += tf

        if term_position > 0:
            weight += term_position * position_in_text(node, docid, index_utils)

        node.weight = weight


def initialize_edges(graph, text_distance, term_embedding, embeddings):
    embeddings_not_found = 0
    for node_key in graph.nodes.keys():
        for other_node_key in graph.nodes.keys():
            if node_key == other_node_key:
                continue

            weight = 0.0
            if text_distance > 0:
                distance = closest_distance(
                    graph.nodes[node_key], graph.nodes[other_node_key]
                )
                weight += text_distance * distance_in_text(distance)

            if term_embedding > 0:
                weight += term_embedding * edge_embedding_weight(
                    graph.nodes[node_key],
                    graph.nodes[other_node_key],
                    embeddings,
                    embeddings_not_found,
                )

            if weight > 0.0:
                graph.add_edge(node_key, other_node_key, weight)
