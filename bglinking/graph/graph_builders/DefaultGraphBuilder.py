import json
import logging

import numpy as np
from bglinking.database_utils import db_utils as db_utils
from bglinking.general_utils import utils
from bglinking.graph.Node import Node


class DefaultGraphBuilder:
    def build(self, **kwargs):
        # Retrieve named entities from database.
        if kwargs["use_entities"]:
            add_named_entities_to_graph(
                kwargs["graph"], kwargs["cursor"], kwargs["docid"]
            )

        # Retrieve top n tfidf terms from database.
        if kwargs["nr_terms"] > 0:
            add_terms_to_graph(
                kwargs["graph"], kwargs["cursor"], kwargs["docid"], kwargs["nr_terms"]
            )

        # Set node weighs
        calculate_node_weights(
            kwargs["graph"],
            kwargs["docid"],
            kwargs["index_utils"],
            kwargs["term_tfidf"],
            kwargs["term_position"],
        )

        # Normalize node weights: enity (tf) and terms (tfidf).
        normalize_node_weights(kwargs["graph"])

        # Create edges and set edge weights
        initialize_edges(
            kwargs["graph"],
            kwargs["text_distance"],
            kwargs["term_embedding"],
            kwargs["embeddings"],
        )


# ----- Start building graph -----
def add_named_entities_to_graph(graph, cursor, docid):
    """Retrieves named entities from the database and adds them to the graph."""
    # Retrieve named entities for current docid
    entities = db_utils.get_entities_from_docid(cursor, docid, "entity_ids")

    # Create node for each entity
    for entity in entities:  # [['Washington Redskins', '[30]', '1', 'ORG'], ..]
        entity_name = entity[0]
        entity_tf = int(entity[2])
        entity_type = entity[3]

        # Convert list string to list (fails occasionally)
        try:
            ent_positions = json.loads(entity[1].replace(";", ""))
        except Exception as e:
            print(f"issue with enitity: {entity[1]}, docid: {docid}")
            logging.info(e)
            continue

        # Add node to graph
        graph.add_node(Node(entity_name, entity_type, ent_positions, entity_tf))


def add_terms_to_graph(graph, cursor, docid, nr_terms):
    """Adds n tfidf terms to the graph."""
    # Retrieve n terms from database
    terms = db_utils.get_entities_from_docid(cursor, docid, "tfidf_terms")[:nr_terms]

    # Create node for each term
    for term in terms:
        term_name = term[0]
        term_positions = json.loads(term[1])
        term_tf = int(term[2])
        graph.add_node(Node(term_name, "term", term_positions, term_tf))


# ----- End building graph -----

# ----- Normalize node weights -----
def normalize_node_weights(graph):
    """Make entity(tf) & term(tf-idf) weights comparable in magnitude."""
    term_weights = {"dummy0000": 0.0}
    entity_weights = {"dummy0000": 0.0}

    # Distinguish between terms and entities.
    for node in graph.nodes.values():
        if node.node_type == "term":
            term_weights[node] = node.weight
        else:
            entity_weights[node] = node.weight

    # Normalize both individually (weights between 0-1).
    normalized_term_weights = utils.normalize_dict(term_weights)
    normalized_entity_weights = utils.normalize_dict(entity_weights)

    # Delete dummies.
    del normalized_term_weights["dummy0000"]
    del normalized_entity_weights["dummy0000"]

    # Write normalized weights to graph.
    for node in graph.nodes.values():
        if node.node_type == "term":
            node.weight = normalized_term_weights[node]
        else:
            node.weight = normalized_entity_weights[node]


# ----- Start node weights section -----
def tf_score(node, N) -> float:
    if node.tf == 1:
        return 1 / N
    else:
        return (1 + np.log(node.tf - 1)) / N


def node_weight_tf(index_utils, graph, node, node_name):
    """Return tf(idf) node weight; idf only for terms."""
    tf = tf_score(node, graph.nr_nodes())
    nr_documents = index_utils.stats()["documents"]

    if node.node_type == "term":
        # term --> return tfidf
        filtered_term = utils.clean_NE_term(node_name)
        df = index_utils.get_term_counts(filtered_term, analyzer=None)[0] + 1e-5
        return utils.tfidf(tf, df, nr_documents)
    else:
        # named entity --> return tf
        return tf


def node_weight_position(node: Node, docid: str) -> float:
    # location refers to the earliest paragraph index (starts at 0, so +1).
    return 1 / (min(node.locations) + 1)


def calculate_node_weights(graph, docid, index_utils, term_tfidf, term_position):
    for node_name, node in graph.nodes.items():
        # Set weight to 0.0 to be sure.
        node.weight = 0.0

        # (1) tf(idf) weight; only idf for terms.
        if term_tfidf > 0:
            node.weight += term_tfidf * node_weight_tf(
                index_utils, graph, node, node_name
            )

        # (2) position weight.
        if term_position > 0:
            node.weight += term_position * node_weight_position(node, docid)


# ----- End node weights section -----


# ----- Start edge initialization & weights section -----
def closest_distance(node_a, node_b):
    """Calculate closest distance between text locatios."""
    min_distance = 999999
    for loc_a in node_a.locations:
        for loc_b in node_b.locations:
            distance = abs(loc_a - loc_b)
            if distance < min_distance:
                min_distance = distance
    return min_distance


def distance_in_text(distance: int) -> float:
    if distance <= 1:
        return 1 / (1.0 + distance)
    else:
        return 0.0


def term_similarity(node_a, node_b, embeddings):
    """Calculate similarity between terms in word embedding."""
    try:
        similarity = embeddings.similarity(node_a.__str__(), node_b.__str__())
    except Exception as e:
        # If term(s) does not occur in embedding similarity is always 0.
        similarity = 0
        logging.info(e)
    return similarity


def initialize_edges(graph, text_distance, term_embedding, embeddings):
    """Initialize edges based on weight function(s)."""
    # For each node check connection with other nodes.
    for node_key in graph.nodes.keys():
        for other_node_key in graph.nodes.keys():
            weight = 0.0
            node_a = graph.nodes[node_key]
            node_b = graph.nodes[other_node_key]

            # Skip self connection.
            if node_key == other_node_key:
                continue

            # (1) Edges based on text distance.
            if text_distance > 0:
                distance = closest_distance(node_a, node_b)
                weight += text_distance * distance_in_text(distance)

            # (2) Edges based on term embeddings.
            if term_embedding > 0:
                similarity = term_similarity(node_a, node_b, embeddings)
                weight += term_embedding * similarity

            # Add edge if edge weight is above threshold.
            edge_threshold = 0.0
            if weight > edge_threshold:
                graph.add_edge(node_key, other_node_key, weight)


# ----- End edge initialization & weights section -----
