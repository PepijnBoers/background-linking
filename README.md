# background-linking
Code to replicate my experiments for background linking in TREC News


## Docker
After cloning the repository build the docker image using the dockerfile:

```
docker build . -t blimg
```

## Resources
- index: Index of the Washington Post Corpus (v2 or v3)
- db: Database with terms and named entities for all topic/candidate docs
- embeddings: Word embedding file
- topics: File with topics (TREC format)
- qrels: Query relevance file for the specified topics
- candidates: Candidate documents

### Index
TREC's [Washington Post](https://trec.nist.gov/data/wapost/) index was build using [Anserini](https://github.com/castorini/anserini), see [Regressions for TREC 2019 Background Linking](https://github.com/castorini/anserini/blob/master/docs/regressions-backgroundlinking19.md). In order to obtain the corpus, an individual agreement form has to be completed first. The exact command we used is shown bellow (note that we used version 2 of the corpus for the 2019 topics, and version 3 for the 2020 topics):

```
./target/appassembler/bin/IndexCollection -collection WashingtonPostCollection \
 -input /WashingtonPost.v2/data -generator WashingtonPostGenerator \
 -index lucene-index.core18.pos+docvectors+rawdocs_all \
 -threads 1 -storePositions -storeDocvectors -storeRaw -optimize -storeContents
```

The obtained index should be stored in `bglinking/resources/Index`.

### Database
A database was created to speed up the graph generation. Named entities and tf-idf terms were stored per candidate document in a database. [REL](https://github.com/informagi/REL) was used for the extraction of named entities, see build_db.py. 

The database should be stored in `bglinking/resources/db`.

### Candidates
Candidates were obtained using BM25 + RM3 via Anserini, see [Regressions for TREC 2019 Background Linking](https://github.com/castorini/anserini/blob/master/docs/regressions-backgroundlinking19.md).

The candidates file should be stored in `bglinking/resources/candidates`

### Embeddings
We made use of embeddings from [GEEER](https://github.com/informagi/GEEER), they can be downloaded from [this](https://surfdrive.surf.nl/files/index.php/s/V2mc4zrcE46Ucvs) link.

The embeddings should be extracted and stored in `bglinking/resources/embeddings`.

### Topics and Qrels
Topics and query relevance files can be downloaded from the News Track [page](https://trec.nist.gov/data/news2019.html).

Store in `bglinking/resources/topics-and-qrels`


## graph configuration
- nr-terms: Number of terms used in the graph (default = 100)
- term-tfidf: Scaler for tf-idf weight of node (default = 1.0)
- term-postition: Scaler for position weight of node (default = 0.0)
- term-embedding: Scaler for embedding weight of edges, i.e. cosine similarity term vectors (default = 0.0)
- text-distance: Scaler for distance weight for edges (default = 0.0)

- use-entities: Append named entities to graph nodes
- textrank: Apply textrank to current graph


## output
- output: Name of output file
- run-tag: Run-tag in output file

## other
- diversity: Apply diversity filter
- stats: Show index stats
- year: Year of TREC edition

# Experiments Graph Configurations
Results are stored in `bglinking/resources/output`

## Graph [100 terms, no edges]
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --nr-terms 100 --output simple_graph_19.txt --run-tag simple_graph
```

## Graph [100 terms, edges based on text distance]
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --nr-terms 100 --text-distance 1 --output simple_graph_text_distance_19.txt \
           --run-tag simple_graph_text_distance
```

## Graph [100 terms, edges based on word embeddings]
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-embedding 1 --output simple_graph_term_embedding_19.txt \
           --run-tag simple_graph_term_embedding
```

## Graph [100 terms - weights based on term position]
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --nr-terms 100 --term-position 1 --output simple_graph_term_position_19.txt \
           --run-tag simple_graph_term_position
```

## Graph configurations combining: term position, text distance & word embedding.
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --nr-terms 100 --term-position 1 --text-distance 1 \
           --output simple_graph_term_position_text_rank_19.txt \
           --run-tag simple_graph_term_position_text_rank

docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 \
           --output simple_graph_term_position_text_embedding_19.txt \
           --run-tag simple_graph_term_position_text_embedding

docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --text-distance 1 --term-embedding 1 \
           --output simple_graph_term_position_text_distance_19.txt \
           --run-tag simple_graph_term_position_text_distance
```

## Add named entities to graph nodes (simplest configuration)
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --use-entities --output only_entities_19.txt --run-tag only_entities
```

## Add named entities to graph nodes (best performing run)
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 --use-entities \
           --output best_graph_entities_19.txt \
           --run-tag best_graph_entities
```

## Test effect of novelty algorithm (without named entities)
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 --novelty 0.05 \
           --output best_graph_novelty_19.txt \
           --run-tag best_graph_novelty
```

## Test effect of novelty algorithm (with named entities)
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 --use-entities --novelty 0.05 \
           --output best_graph_novelty_entities_19.txt --run-tag best_graph_novelty_entities
```

## Test best run with TextRank algorithm
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 --textrank \
           --output best_graph_textrank_19.txt \
           --run-tag best_graph_textrank
```

## Test diversification
```
docker run --rm -v $PWD/bglinking/resources:/opt/background-linking/bglinking/resources blimg \
           --index lucene-index.core18.pos+docvectors+rawdocs_all --db entity_database_19.db \
           --topics topics.backgroundlinking19.txt --qrels qrels.backgroundlinking19.txt \
           --candidates run.backgroundlinking19.bm25+rm3.topics.backgroundlinking19.txt \
           --embedding WKN-vectors/WKN-vectors.bin \
           --nr-terms 100 --term-position 1 --term-embedding 1 --diversify --use-entities \
           --output best_graph_diversified_19.txt --run-tag best_graph_diversified
```