"""Generate predictions jsonl file given input tsv file."""
import argparse
import json
import numpy as np
import os
import re
import sys
import tensorflow as tf
from tqdm import tqdm

from docqa.data_processing.document_splitter import Truncate, TopTfIdf
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable, CachingResourceLoader, ResourceLoader

from util import *

OPTS = None

DEFAULT_BEAM_SIZE = 100

def parse_args():
  parser = argparse.ArgumentParser('Generate predictions for a batch of inputs.')
  parser.add_argument('model')
  parser.add_argument('input_file', metavar='input.tsv')
  parser.add_argument('output_file', metavar='output.jsonl')
  parser.add_argument('--beam-size', '-k', type=int, default=DEFAULT_BEAM_SIZE,
                      help='Beam size')
  parser.add_argument('--no-vec', action='store_true')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def read_input_data(model):
  data = []
  vocab = set()
  tokenizer = NltkAndPunctTokenizer()
  splitter = Truncate(400)  # NOTE: we truncate past 400 tokens
  selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
  with open(OPTS.input_file) as f:
    for i, line in enumerate(f):
      try:
        document_raw, question_raw = line.strip().split('\t')
      except ValueError as e:
        print(line.strip())
        print('Error at line %d' % i)
        raise e
      document = re.split("\s*\n\s*", document_raw)
      question = tokenizer.tokenize_paragraph_flat(question_raw)
      doc_toks = [tokenizer.tokenize_paragraph(p) for p in document]
      split_doc = splitter.split(doc_toks)
      context = selector.prune(question, split_doc)
      if model.preprocessor is not None:
        context = [model.preprocessor.encode_text(question, x) for x in context]
      else:
        context = [flatten_iterable(x.text) for x in context]
      vocab.update(question)
      for txt in context:
        vocab.update(txt)
      ex = [ParagraphAndQuestion(x, question, None, "user-question%d"%i)
            for i, x in enumerate(context)]
      data.append((document_raw, question_raw, context, ex))
  return data, vocab

def main():
  print('Starting...')
  model_dir = ModelDir(OPTS.model)
  model = model_dir.get_model()
  if not isinstance(model, ParagraphQuestionModel):
    raise ValueError("This script is built to work for ParagraphQuestionModel models only")
  input_data, vocab = read_input_data(model)

  print('Loading word vectors...')
  model.set_input_spec(ParagraphAndQuestionSpec(batch_size=None), vocab)

  print('Starting Tensorflow session...')
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    prediction = model.get_prediction()
    # Take 0-th here because we know we only truncate to one paragraph
    start_logits_tf = prediction.start_logits[0]
    end_logits_tf = prediction.end_logits[0]
    none_logit_tf = prediction.none_logit[0]
    context_rep_tf = model.context_rep[0]
    m1_tf = model.predictor.m1[0]
    m2_tf = model.predictor.m2[0]
  model_dir.restore_checkpoint(sess)

  with open(OPTS.output_file, 'w') as f:
    for doc_raw, q_raw, context, ex in tqdm(input_data):
      encoded = model.encode(ex, is_train=False)
      start_logits, end_logits, none_logit, context_rep, m1, m2 = sess.run(
          [start_logits_tf, end_logits_tf, none_logit_tf, context_rep_tf,
           m1_tf, m2_tf],
          feed_dict=encoded)
      beam, p_na = logits_to_probs(
          doc_raw, context[0], start_logits, end_logits, none_logit,
          beam_size=OPTS.beam_size)
      inputs = [context_rep, m1, m2]
      vec = np.concatenate([np.amax(x, axis=0) for x in inputs] +
                           [np.amin(x, axis=0) for x in inputs] +
                           [np.mean(x, axis=0) for x in inputs])
      #span_logits = np.add.outer(start_logits, end_logits)
      #all_logits = np.concatenate((np.array([none_logit]), span_logits.flatten()))
      #log_partition = scipy.special.logsumexp(all_logits)
      #vec = np.concatenate([
      #    np.amax(context_rep, axis=0),
      #    np.amin(context_rep, axis=0),
      #    np.mean(context_rep, axis=0),
      #    [np.amax(start_logits), scipy.special.logsumexp(start_logits),
      #     np.amax(end_logits), scipy.special.logsumexp(end_logits),
      #     none_logit, log_partition] 
      #])
      out_obj = {'paragraph': doc_raw, 'question': q_raw,
                 'beam': beam, 'p_na': p_na}
      if not OPTS.no_vec:
        out_obj['vec'] = vec.tolist()
      print(json.dumps(out_obj), file=f)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
