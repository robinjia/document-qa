"""Extract distant negatives for SQuAD."""
import argparse
import collections
import json
import numpy as np
import os
import sys

from docqa.data_processing.preprocessed_corpus import PreprocessedData, preprocess_par
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.squad.squad_data import SquadCorpus
from docqa.squad.squad_document_qa import SquadTfIdfRanker

SQUAD_JSON_DIR = 'squad_json_data'

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('split', choices=['train', 'dev'])
  parser.add_argument('--out-file', '-o')
  parser.add_argument('--num-per-orig', '-n', type=int, default=4)
  return parser.parse_args()

def write_output(split, new_data, out_file):
  orig_filename = os.path.join(SQUAD_JSON_DIR, '%s-v1.1.json' % split)
  with open(orig_filename) as f:
    out_data = json.load(f)
  qid_to_question = {}
  for doc in out_data['data']:
    for p in doc['paragraphs']:
      for qa in p['qas']:
        qid_to_question[qa['id']] = qa['question']
  num_neg = 0
  neg_examples = collections.defaultdict(list)
  for mpq in new_data:
    for i, p in enumerate(mpq.paragraphs):
      if all(a not in p.original_text for a in mpq.answer_text):
        neg_examples[p.original_text].append((mpq.question_id, i))
        num_neg += 1
  print('%d negatives without original answer' % num_neg)
  num_neg_added = 0
  for doc in out_data['data']:
    for p in doc['paragraphs']:
      for qid, index in neg_examples[p['context']]:
        num_neg_added += 1
        p['qas'].append({
            'id': '%s_neg%d' % (qid, index),
            'question': qid_to_question[qid],
            'answers': []
        })
  print('%d negatives added (should match other number)' % num_neg_added)
  with open(out_file, 'w') as f:
    json.dump(out_data, f)


def main():
  corpus = SquadCorpus()
  prepro = SquadTfIdfRanker(NltkPlusStopWords(True), OPTS.num_per_orig, True)
  orig_data = corpus.get_train() if OPTS.split == 'train' else corpus.get_dev()
  orig_lens = [len(p.text[0]) for doc in orig_data for p in doc.paragraphs
               for q in p.questions] 
  new_data = preprocess_par(orig_data, corpus.evidence, prepro, n_processes=1)
  new_lens = [len(p.text) for q in new_data for p in q.paragraphs]
  print('%d original, mean %.2f words' % (len(orig_lens), np.mean(orig_lens)))
  print('%d new, mean %.2f words'% (len(new_lens), np.mean(new_lens)))
  if OPTS.out_file:
    write_output(OPTS.split, new_data, OPTS.out_file)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
