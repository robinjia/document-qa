import argparse
import bottle
import numpy as np
import os
import re
import sys
import tensorflow as tf

from docqa.data_processing.document_splitter import Truncate, TopTfIdf
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable, CachingResourceLoader, ResourceLoader

from util import logits_to_probs

OPTS = None

MAX_SPAN_LENGTH = 8
BEAM_SIZE = 10

def parse_args():
  parser = argparse.ArgumentParser('Start a demo server for QA over a single document.')
  parser.add_argument('model', help='Model directory')
  parser.add_argument('--hostname', '-n', default='0.0.0.0', help='Hostname')
  parser.add_argument('--port', '-p', default=5000, help='Port number')
  parser.add_argument('--debug', '-d', action='store_true', help='Run in debug mode')
  parser.add_argument('--reload-vocab',  action='store_true', 
                      help='Reload word vectors each time (faster startup, higher latency)')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  print('Starting...')
  model_dir = ModelDir(OPTS.model)
  model = model_dir.get_model()
  tokenizer = NltkAndPunctTokenizer()
  if not isinstance(model, ParagraphQuestionModel):
      raise ValueError("This script is built to work for ParagraphQuestionModel models only")
  if OPTS.reload_vocab:
    loader = ResourceLoader()
  else:
    loader = CachingResourceLoader()
  print('Loading word vectors...')
  model.set_input_spec(ParagraphAndQuestionSpec(batch_size=None), set([',']),
                       word_vec_loader=loader, allow_update=True)
  print('Starting Tensorflow session...')
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    prediction = model.get_prediction()
    # Take 0-th here because we know we only truncate to one paragraph
    start_logits_tf = prediction.start_logits[0]
    end_logits_tf = prediction.end_logits[0]
    none_logit_tf = prediction.none_logit[0]
    #best_spans_tf, conf_tf = prediction.get_best_span(MAX_SPAN_LENGTH)
  model_dir.restore_checkpoint(sess)
  splitter = Truncate(400)  # NOTE: we truncate past 400 tokens
  selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
  app = bottle.Bottle()

  @app.route('/')
  def index():
    return bottle.template('index')

  @app.route('/post_query', method='post')
  def post_query():
    document_raw = bottle.request.forms.getunicode('document').strip()
    question_raw = bottle.request.forms.getunicode('question').strip()
    document = re.split("\s*\n\s*", document_raw)
    question = tokenizer.tokenize_paragraph_flat(question_raw)
    doc_toks = [tokenizer.tokenize_paragraph(p) for p in document]
    split_doc = splitter.split(doc_toks)
    context = selector.prune(question, split_doc)
    if model.preprocessor is not None:
      context = [model.preprocessor.encode_text(question, x) for x in context]
    else:
      context = [flatten_iterable(x.text) for x in context]
    vocab = set(question)
    for txt in context:
      vocab.update(txt)
    data = [ParagraphAndQuestion(x, question, None, "user-question%d"%i)
            for i, x in enumerate(context)]
    model.word_embed.update(loader, vocab)
    encoded = model.encode(data, is_train=False)
    start_logits, end_logits, none_logit = sess.run(
        [start_logits_tf, end_logits_tf, none_logit_tf], feed_dict=encoded)
    beam, p_na = logits_to_probs(
        document_raw, context[0], start_logits, end_logits, none_logit,
        beam_size=BEAM_SIZE)
    return bottle.template('results', document=document_raw, question=question_raw, 
                           beam=beam, p_na=p_na)

  cur_dir = os.path.abspath(os.path.dirname(__file__))
  bottle.TEMPLATE_PATH.insert(0, os.path.join(cur_dir, 'views'))
  bottle.run(app, host=OPTS.hostname, port=OPTS.port, debug=OPTS.debug)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
