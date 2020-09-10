import os
import argparse
import requests
import sqlite3

from bglinking.general_utils import utils
from bglinking.database_utils import db_utils
from bglinking.database_utils.create_db import create_db

from pyserini import index
from pyserini.search import get_topics

from tqdm import tqdm
from collections import defaultdict


# REL Info
IP_ADDRESS = "https://rel.cs.ru.nl/api"


def get_docids(year: str, topics_only: bool) -> list:
    print(f'type: {type(year)}, {year}, {year==20}')
    res_file = f'../output/runs/run.backgroundlinking{year}.bm25+rm3.topics.backgroundlinking{year}.txt'
    qid_docids = utils.read_docids_from_file(res_file)
    if year == 20:
        topic_docids = utils.read_topic_ids_from_file('../resources/topics-and-qrels/topics.backgroundlinking20.txt')
    else:
        topic_docids = [topic['title'] for topic in get_topics(f'trec20{year}_bl').values()]
    
    if topics_only:
        return topic_docids
    else:
        return [docid for qid in qid_docids.keys() for docid in qid_docids[qid]] + topic_docids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', dest='index',
                        default='lucene-index.core18.pos+docvectors+rawdocs_all_v3', help='Document index')
    parser.add_argument('--name', dest='name', default='default_database_name',
                        help='Database name without .db or path')
    parser.add_argument('--extractor', dest='extractor', default='rel',
                        help='Module for entity extraction (rel or spacy)')
    parser.add_argument('--year', dest='year', default='20', type=int,
                        help='Year of TREC News background linking edition')
    parser.add_argument('--topics-only', dest='topics_only', default=False, action='store_true',
                        help='Use only topic ids')
    parser.add_argument('-n', dest='n', default=100, type=int,
                        help='Number of tfidf terms to extract')
    parser.add_argument('--cut', dest='cut', default=9999999, type=int,
                        help='Cut off used to build smaller sample db.')
    args = parser.parse_args()

    # Check if database exists if not, create it:
    if not os.path.exists(f'../resources/db/{args.name}.db'):
        create_db(args.name)

    # Index Utility
    index_utils = index.IndexReader(f'../resources/Index/{args.index}')
    total_docs = index_utils.stats()['non_empty_documents']

    # Docids
    all_docids = get_docids(args.year, args.topics_only)

    # Connect Database
    conn, cursor = db_utils.connect_db(f'../resources/db/{args.name}.db')

    # Loop over docids:
    for docid in tqdm(all_docids[:args.cut]):

        # Extract all paragraphs from doc and store in list.
        contents = index_utils.doc_contents(docid).split('\n')

        # Obtain top n tfidf terms in doc
        tfidf_terms = utils.create_top_n_tfidf_vector(
            index_utils, docid, n=args.n, total_N=total_docs)

        # Keep track of entity/term locations
        location_entities = {}
        term_locations = defaultdict(list)

        # Loop over paragraphs
        for i, content in enumerate(contents):

            # Tfidf terms
            analyzed_terms = index_utils.analyze(content)
            present_terms = list(set(analyzed_terms).intersection(tfidf_terms))
            for term in present_terms:
                term_locations[term].append(i)

            # Rel named entities
            if args.extractor == 'rel':
                document = {"text": content, "spans": [], }
                rel_request = requests.post("{}".format(IP_ADDRESS), json=document).json()
                location_entities[i] = [(entity[3], entity[6]) for entity in rel_request]

        # Format tfidf terms
        terms = [f'{term};;;{locations};;;{tfidf_terms[term]}' 
                for term, locations in term_locations.items()]

        # Format named entities
        entity_loc_dict = defaultdict(list)
        entity_type_dict = {}
        for i, value in location_entities.items():
            for entity_tuple in value:
                entity_name, entity_type = entity_tuple
                entity_loc_dict[entity_name].append(i)
                entity_type_dict[entity_name] = entity_type
        entities = [f'{entity};;;{locations};;;{len(locations)};;;{entity_type_dict[entity]}'
                    for entity, locations in entity_loc_dict.items()]

        # Insert into sql database.
        cursor.execute('INSERT INTO entities (docid, entity_ids, tfidf_terms) VALUES (?,?,?)',
                       (docid, '\n'.join(entities), '\n'.join(terms)))
        conn.commit()

    conn.close()
