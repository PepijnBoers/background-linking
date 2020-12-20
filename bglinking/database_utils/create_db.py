# Database creation file.
import argparse
import sqlite3


def create_db(db_name):
    conn = sqlite3.connect(f"../resources/db/{db_name}.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE "entities" (
        "id"	integer,
        "docid"	text,
        "entity_ids"	text,
        "tfidf_terms"	text,
        PRIMARY KEY("id"))"""
    )
    conn.commit()
    conn.close()


def create_db_dbpedia():
    conn = sqlite3.connect("db/EntityReaderDBpedia.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE "entities" (
        "id"	integer,
        "doc_id"	text,
        "para_id"	text,
        "text"	text,
        "entity_ids"	text,
        PRIMARY KEY("id"))"""
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        dest="name",
        default="default_database_name",
        help="Database name without .db",
    )
    args = parser.parse_args()
    create_db(args.name)
