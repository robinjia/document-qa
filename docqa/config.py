from os.path import join, expanduser, dirname

"""
Global config options
"""

VEC_DIR = 'glove'
SQUAD_SOURCE_DIR = '.'
SQUAD_TRAIN = join(SQUAD_SOURCE_DIR, "train-v1.1.json")
SQUAD_DEV = join(SQUAD_SOURCE_DIR, "dev-v1.1.json")


TRIVIA_QA = join("data", "triviaqa")
TRIVIA_QA_UNFILTERED = join("data", "triviaqa-unfiltered")

DOCUMENT_READER_DB = join("data", "doc-rd", "docs.db")


CORPUS_DIR = 'data'
