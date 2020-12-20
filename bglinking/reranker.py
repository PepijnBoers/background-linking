import argparse
import os

from pyserini import index
from tqdm import tqdm

from bglinking.database_utils import db_utils
from bglinking.general_utils import utils
from bglinking.graph.graph import Graph
from bglinking.graph.graph_comparators.GMCSComparator import GMCSComparator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--index",
    dest="index",
    default="lucene-index.core18.pos+docvectors+rawdocs_all",
    help="specify the corpus index",
)

parser.add_argument(
    "--db", dest="db", default="entity_database_19.db", help="specify the database"
)

parser.add_argument(
    "--embedding", dest="embedding", default="", help="specify the embeddings to use"
)

parser.add_argument("--stats", dest="stats", default=False, help="Show index stats")

parser.add_argument(
    "--year", dest="year", default=None, type=int, help="TREC year 18, 19 or 20"
)

parser.add_argument(
    "--topics",
    dest="topics",
    default="topics.backgroundlinking19.txt",
    help="specify qrels file",
)

parser.add_argument(
    "--candidates",
    dest="candidates",
    default="run.backgroundlinking20.bm25+rm3.topics.backgroundlinking20.txt",
    help="Results file that carries candidate docs",
)

parser.add_argument(
    "--qrels",
    dest="qrels",
    default="qrels.backgroundlinking19.txt",
    help="specify qrels file",
)

parser.add_argument(
    "--output", dest="output", default="output_graph.txt", help="specify output file"
)

parser.add_argument(
    "--run-tag", dest="run_tag", default="unspecified_run_tag", help="specify run tag"
)

parser.add_argument(
    "--anserini",
    dest="anserini",
    default="/Volumes/Samsung_T5/anserini",
    help="path to anserini",
)

parser.add_argument(
    "--textrank",
    dest="textrank",
    default=False,
    action="store_true",
    help="Apply TextRank",
)

parser.add_argument(
    "--use-entities",
    dest="use_entities",
    default=False,
    action="store_true",
    help="Use named entities as graph nodes",
)

parser.add_argument(
    "--nr-terms",
    dest="nr_terms",
    default=0,
    type=int,
    help="Number of tfidf terms to include in graph",
)

parser.add_argument(
    "--term-tfidf",
    dest="term_tfidf",
    default=1.0,
    type=float,
    help="""Weight that should be assigned to tfidf score
            of terms (for node initialization)""",
)

parser.add_argument(
    "--term-position",
    dest="term_position",
    default=0.0,
    type=float,
    help="Weight for term position in initial node weight",
)

parser.add_argument(
    "--term-embedding",
    dest="term_embedding",
    default=0.0,
    type=float,
    help="Weight for word embeddings in edge creation",
)

parser.add_argument(
    "--text-distance",
    dest="text_distance",
    default=0.0,
    type=float,
    help="Weight for text distance in edge creation",
)

parser.add_argument(
    "--l",
    dest="node_edge_l",
    default=0.5,
    type=float,
    help="Weight for importance nodes over edges",
)

parser.add_argument(
    "--novelty",
    dest="novelty",
    default=0.0,
    type=float,
    help="Weight for novelty in relevance score",
)

parser.add_argument(
    "--diversify",
    dest="diversify",
    default=False,
    action="store_true",
    help="Diversify the results according to entity types",
)

args = parser.parse_args()
# utils.write_run_arguments_to_log(**vars(args))

if args.diversify and not args.use_entities:
    parser.error("--diversify requires --use-entities.")

if args.year is not None:
    if args.year == 20:
        args.index = "lucene-index.core18.pos+docvectors+rawdocs_all_v3"
    args.db = f"entity_database_{args.year}.db"
    args.topics = f"topics.backgroundlinking{args.year}.txt"
    args.candidates = (
        f"run.backgroundlinking{args.year}"
        f".bm25+rm3.topics.backgroundlinking{args.year}.txt"
    )
    args.qrels = f"newsir{args.year}-qrels-background.txt"


print(f"\nIndex: resources/Index/{args.index}")
print(f"Topics were retrieved from resources/topics-and-qrels/{args.topics}")
print(f"Results are stored in resources/output/runs/{args.output}\n")
utils.create_new_file_for_sure(f"resources/output/{args.output}")

# '../database_utils/db/rel_entity_reader.db'
conn, cursor = db_utils.connect_db(f"resources/db/{args.db}")

# load word embeddings
if args.term_embedding > 0 and args.embedding != "":
    embeddings = utils.load_word_vectors(f"resources/embeddings/{args.embedding}")
    print("Embeddings sucessfully loaded!")
else:
    embeddings = {}

# Load index
index_utils = index.IndexReader(f"resources/Index/{args.index}")

# Configure graph options.
comparator = GMCSComparator()

# Build kwargs for graph initialization:
build_arguments = {
    "index_utils": index_utils,
    "cursor": cursor,
    "embeddings": embeddings,
    "use_entities": args.use_entities,
    "nr_terms": args.nr_terms,
    "term_tfidf": args.term_tfidf,
    "term_position": args.term_position,
    "text_distance": args.text_distance,
    "term_embedding": args.term_embedding,
}

# Read topics from file
topics = utils.read_topics_and_ids_from_file(
    f"resources/topics-and-qrels/{args.topics}"
)

for topic_num, topic in tqdm(topics):
    query_num = str(topic_num)
    query_id = topic  # ['title']

    query_graph = Graph(query_id, f"query_article_{query_num}")
    query_graph.build(**build_arguments)

    # recalculate node weights using TextRank
    if args.textrank:
        query_graph.rank()

    # Create new ranking.
    ranking = {}
    addition_types = {}

    # Read candidate docs for current topic.
    qid_docids = utils.read_docids_from_file(f"resources/candidates/{args.candidates}")

    # Loop over candidate docs and calculate similarity score per doc.
    for docid in qid_docids[query_num]:
        # Create graph for candidate.
        fname = f"candidate_article_{query_num}_{docid}"
        candidate_graph = Graph(docid, fname)
        candidate_graph.set_graph_comparator(comparator)

        # Build (initialize) graph nodes and edges.
        candidate_graph.build(**build_arguments)

        # Recalculate node weights using TextRank
        if args.textrank:
            candidate_graph.rank()

        # Calculate relevance score + diversity type
        relevance_score, diversity_type = candidate_graph.compare(
            query_graph, args.novelty, args.node_edge_l
        )

        # Store relevance score in dict
        ranking[docid] = relevance_score

        # Store diversity type in dict
        addition_types[docid] = diversity_type

    # Sort retrieved documents according to new similarity score.
    sorted_ranking = utils.normalize_dict(
        {
            k: v
            for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)
        }
    )

    # Diversify results (filter)
    if args.diversify:
        to_delete_docids = utils.diversity_filter(addition_types, sorted_ranking)
        for key in to_delete_docids[:85]:  # delete max 85 documents per topic.
            del sorted_ranking[key]

    # Store results in txt file.
    utils.write_to_results_file(
        sorted_ranking, query_num, args.run_tag, f"resources/output/{args.output}"
    )

if args.year != 20:
    # Evaluate performance with trec_eval.
    os.system(
        (
            "/opt/anserini-tools/eval/trec_eval.9.0.4/trec_eval -c -M1000 -m map"
            f" -m ndcg_cut -m P.10 resources/topics-and-qrels/{args.qrels}"
            f" resources/output/{args.output}"
        )
    )
