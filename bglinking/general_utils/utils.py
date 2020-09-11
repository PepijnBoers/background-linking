import json
import numpy as np
import re
import math
import time

from operator import attrgetter
from scipy import stats
from ast import literal_eval
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import Callable

from bglinking.database_utils import db_utils
from bglinking.general_utils.str_to_dict import turn_into_dict

from pyserini import analysis
from pyserini.analysis import get_lucene_analyzer

import gensim


def timer(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"function: {str(func)} took {end-start:.4f} sec.")
        return result
    return wrapper


def limit_set(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        n = 5
        sorted_results = {k: v for k, v in sorted(
            results.items(), key=lambda item: item[1], reverse=True)[:n]}
        return sorted_results
    return wrapper


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / (union+0.0001)


def jaccard_similarity_weighted(list1, list2, w_dict1, w_dict2):
    intersection = list(set(list1).intersection(list2))
    weighted_intersection = np.sum(
        [w_dict1[node]*w_dict2[node] for node in intersection])
    union = (len(list1) + len(list2)) - weighted_intersection
    return float(weighted_intersection) / (union+0.0001)


def prevent_zero(i):
    if i == 0:
        return 0.000000001
    else:
        return i


def normalize_dict(dictionary):
    max_rank = np.max(list(dictionary.values()))
    min_rank = np.min(list(dictionary.values()))
    for n, w in dictionary.items():
        dictionary[n] = (w - min_rank) / prevent_zero((max_rank - min_rank))
    return dictionary


def standardize_dict(dictionary):
    mean = np.mean(list(dictionary.values()))
    std = np.std(list(dictionary.values()))
    for n, w in dictionary.items():
        dictionary[n] = (w - mean) / prevent_zero(std)
    return dictionary


def get_article_text(raw_doc, stemming=False) -> str:
    """Retrieve actual text from raw document (as obtained via Pyserini)."""
    res = turn_into_dict(raw_doc)
    contents = res['contents']  # list of dictrionaries
    text = ''

    for content in contents:
        try:
            if "subtype" in content.keys() and content["subtype"] == "paragraph":
                text += content["content"]+'\n'
        except AttributeError:
            continue

    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(" ", strip=True)


def get_article_title(raw_doc, stemming=False) -> str:
    res = turn_into_dict(raw_doc)
    return res['title']


def show_article(index_utils, docid, relevance='unknown', print_article=True, n_chars=500, return_text=False):
    raw_doc = index_utils.doc_raw(docid)
    text = get_article_text(raw_doc, stemming=False)
    title = get_article_title(raw_doc, stemming=False)
    if print_article:
        print(f"---------------------------------")
        print(f"docid: {docid}\n")
        print(f"relevance: {relevance}\n")
        print(f"Title: {title}\n")
        print(f"Text: \n{text[:n_chars]}...")
        print(f"---------------------------------")

    if return_text:
        return text


def is_valid_article(raw_doc, score):
    res = json.loads(raw_doc)
    try:
        kicker = res['contents'][0]['content']
    except KeyError as e:
        kicker = "dummy"
    return (kicker not in ['Opinions', 'Letters to the Editor', 'The Post\'s View']) and score >= 0


def create_new_file_for_sure(file_name):
    """Temporary solution to make sure the file is newly created."""
    results_file = open(file_name, 'w')
    results_file.close()


def resolve_tie(score, rank, last_score, dup):
    score = round(score, 6)
    if rank == 0 or (last_score - score) > 10**(-4):
        dup = 0
    else:
        dup += 1
        score -= 10**(-6) * dup
    return score, dup


def write_to_results_file(ranking: dict, query_num: str, run_tag: str, file_name: str, last_score=True):
    """Write results to txt."""
    last_score = 0
    dup = 0
    results_file = open(file_name, 'a')
    i = 0
    for rank, docid in enumerate(list(ranking.keys())):
        last_score, dup = resolve_tie(ranking[docid], rank, last_score, dup)
        results_file.write(
            f"{query_num} Q0 {repair_docid(docid)} {i+1} {last_score} {run_tag}\n")
        i += 1
    results_file.close()


def create_top_n_bm25_query(index_utils, docid, n):
    """Generate a query of top n terms with highest bm25 term weights."""
    doc_vec = index_utils.get_document_vector(docid)
    bm_25_per_term = {term: index_utils.compute_bm25_term_weight(
                      docid=docid, term=term, analyzer=None) for term in doc_vec.keys()}
    bm_25_terms_sorted = [term for term, bm25 in sorted(
                          bm_25_per_term.items(), key=itemgetter(1), reverse=True)]
    return " ".join(bm_25_terms_sorted[:n])


def tfidf(tf, df, N):
    return math.log((1.0 + N)/df)*tf


def bm25(idf: float, tf: float, k=0.5, b=0.5, dlen=1.0, avg_dlen=1.0) -> float:
    return idf * (tf*(k+1)/(tf + (k*(1-b+b*(dlen/avg_dlen)))))


def clean_NE_term(term: str) -> str:
    term = re.sub(r'\([^)]*\)', '', term)
    term = re.sub(r'[,_]+', ' ', term)
    filtered_term = re.sub(r'[ \t]+$', '', term)
    return filtered_term


def calc_tfidf_term(term: str, docid: str, index_utils, ne_tf_dict, graph_list) -> float:
    """Calculate tf*idf score for unanalyzed term in doc"""
    # Check if term is a named entity.
    if term in ne_tf_dict.keys():
        tf = ne_tf_dict[term]
        df = named_entity_df(graph_list, term)
    else:
        # tfidf terms are already analyzed.
        analyzed_term = term
        tf = index_utils.get_document_vector(docid)[analyzed_term]
        df = index_utils.get_term_counts(analyzed_term, analyzer=None)[0]

    return tfidf(tf, df)


def create_top_n_tfidf_vector(index_utils, docid: str, n: int, t: int, total_N=595031) -> dict:
    """Create list of top N terms with highest tfidf in a document accompanied with their tf."""
    # retrieve already analyzed terms in dict: tf
    tf = index_utils.get_document_vector(docid)

    # Filter terms: should not contain numbers and len >= 2.
    w_pattern = re.compile("[a-z]+")
    filtered_tf = {term: tf for term, tf in tf.items() if len(w_pattern.findall(term)) == 1 and
                   len(term.replace('.', '')) >= 2 and
                   re.search("[a-z]+", term)[0] == term}

    # df
    df = {term: (index_utils.get_term_counts(term, analyzer=None))
          [0] for term in filtered_tf.keys()}

    # calcute tfidf for each term and store in dict.
    terms_tfidf = {term: tfidf(tf[term], df[term], total_N) for term in filtered_tf.keys()
                   if tfidf(tf[term], df[term], total_N) >= t}

    # Sort terms based on tfidf score.
    tfidf_terms_sorted = {term: tf[term] for term, tfidf in sorted(
        terms_tfidf.items(), key=itemgetter(1), reverse=True)[:n]}

    return tfidf_terms_sorted


def get_tfidf_terms_text(text, index_utils, n, total_N=595031):
    """Extract tf idfterms from a text with wapo as background corpus."""
    # retrieve already analyzed terms in dict: tf
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    analyzed_text = analyzer.analyze(text)
    unique, counts = np.unique(analyzed_text, return_counts=True)
    tf = dict(zip(unique, counts))

    # Filter terms: should not contain numbers and len >= 2.
    w_pattern = re.compile("[a-z]+")
    filtered_tf = {term: tf for term, tf in tf.items() if len(w_pattern.findall(term)) == 1 and
                   len(term.replace('.', '')) >= 2 and
                   re.search("[a-z]+", term)[0] == term}

    # df
    df = {term: (index_utils.get_term_counts(term, analyzer=None))
          [0] for term in filtered_tf.keys()}

    # calcute tfidf for each term and store in dict.
    terms_tfidf = {term: tfidf(tf[term], df[term], total_N) for term in filtered_tf.keys()
                   if tfidf(tf[term], df[term], total_N) >= 3.5}

    # Sort terms based on tfidf score.
    tfidf_terms_sorted = {term: tf[term] for term, tfidf in sorted(
        terms_tfidf.items(), key=itemgetter(1), reverse=True)[:n]}

    return tfidf_terms_sorted


def find_content_parts_for_term(term, contents):
    '''give analyzed term and analyzed content parts'''
    return [i for i, part in enumerate(contents) if term in part]


def create_top_n_tfidf_query(index_utils, docid, n, total_N):
    tfidf_terms_sorted_filtered = create_top_n_tfidf_vector(
        index_utils, docid, n, total_N)
    return " ".join(tfidf_terms_sorted_filtered[:n])


def load_results(path: str) -> dict:
    results = {}
    for line in open(path):
        line_content = line.split(" ")
        qid = line_content[0]
        docid = line_content[2]
        score = line_content[4]

        try:
            results[qid].update({docid: score})
        except KeyError:
            results[qid] = {docid: score}

    return results  # {qid: {docid: score, docid2: score, ..}}


def entity_to_query(entity: str) -> str:
    entity = re.sub(r'\([^)]*\)', '', entity)
    return re.sub(r'[_]+', ' ', entity).rstrip()


def write_run_arguments_to_log(**kwargs):
    run_name = kwargs['output']
    with open(f'resources/output/logs/{run_name}', 'w+') as f:
        for arg, value in kwargs.items():
            f.write(f'{arg}: {value}\n')


def load_word_vectors(embedding_path):
    return gensim.models.KeyedVectors.load(embedding_path, mmap='r')


def word_vector(word: str, embedding: dict) -> list:
    term = word.lower()
    if term in embedding:
        return embedding[term]
    else:
        return np.zeros(50)


def repair_docid(docid: str):
    return re.sub('_PASSAGE[0-9]+', '', docid)


def read_docids_from_file(file_path: str) -> list:
    docids_per_qid = defaultdict(list)
    with open(file_path) as f:
        content = f.readlines()
        lines = [line.split(' ') for line in content]

        for line in lines:
            docids_per_qid[line[0]].append(line[2])

    return docids_per_qid


def read_topic_ids_from_file(file_path: str):
    with open(file_path) as f:
        content = f.readlines()
        topics = [line.strip()[7:-8] for line in content if line.strip(
        ).startswith('<docid>') and line.strip().endswith('</docid>')]
        return topics


def read_topics_and_ids_from_file(file_path: str):
    topics = []
    with open(file_path) as f:
        content = f.read()
        soup = BeautifulSoup(content, features="lxml")
        for topic in soup.find_all('top'):
            num_string = topic.find('num').string
            num = [int(s) for s in num_string.split() if s.isdigit()]
            docid = topic.find('docid').string
            assert len(num) == 1, 'Found more topic numbers for single topic..'
            topics.append([num[0], docid])
    return topics


def not_in_list_2(l1: list, l2: list) -> list:
    return [elem for elem in l1 if elem not in l2]


def get_keys_max_values(dictionary: dict):
    # Get rid off term key
    if 'term' in dictionary.keys():
        del dictionary['term']

    # Find keys with highest value
    max_types = []
    max_value = -1
    for key, value in dictionary.items():
        if value == max_value:
            max_types.append(key)
        if value > max_value:
            max_types = [key]
            max_value = value
    return max_types
