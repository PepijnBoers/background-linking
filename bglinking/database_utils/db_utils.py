import ast
import sqlite3


def connect_db(db_name: str):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    return conn, cursor


def get_docids(cursor) -> list:
    """Retrieve all document ids from database."""
    cursor.execute("SELECT DISTINCT doc_id FROM entities")
    docs = cursor.fetchall()
    if len(docs) > 0:
        return list(docs[0])
    else:
        return []


def get_entities_from_docid(cursor, docid: str, column: str) -> list:
    """Get entity list for specific document from database."""
    cursor.execute(f"SELECT {column} FROM entities WHERE docid=?", (docid,))
    entity_tuple_list = cursor.fetchall()
    entity_list = []
    for tup in entity_tuple_list:
        for entity in tup[0].split("\n"):
            # Some times a '' is found in the splitted string.
            if entity == "":
                continue
            entity_list.append(entity.split(";;;"))

    # Always return a list of lists
    if len(entity_list) == 0:
        print(docid)
        print("zero")
        return []
    else:
        return entity_list


def get_tfidf_terms_from_docid(cursor, doc_id: str) -> list:
    cursor.execute("SELECT tfidf_terms FROM entities WHERE docid=?", (doc_id,))
    entity_tuple_list = cursor.fetchall()
    try:
        tfidf_terms = entity_tuple_list[0][0]
    except IndexError:
        tfidf_terms = "[]"
    return ast.literal_eval(tfidf_terms)


def add_entities_to_docid(c, docid, para_id, text, entity_ids):
    assert 1 == 2, print("wordt dit ooit gebruikt? add_entities_to_docid(...)")
    c.execute(
        "INSERT INTO entities (doc_id, para_id, text, entity_ids) VALUES (?,?,?,?)",
        (docid, para_id, text, entity_ids),
    )
