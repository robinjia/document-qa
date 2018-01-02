import numpy as np
import scipy

def get_preimages(tok):
  """Get possible pre-images of token.

  Necessary because of clean_text() inside NltkAndPunctTokenizer
  inside docqa.data_processing.text_utils
  """
  if tok == '\"':
    return ["''", '``', '\"']
  elif tok == '-':
    return ['\u2212', '-']
  elif tok == '\u2013':
    return ['\u2014', '\u2013']
  return [tok]

def make_tok_to_char(document, tokens):
  tok_to_char = []
  i = 0
  for t in tokens:
    preimages = get_preimages(t)
    while True:
      matches = [x for x in preimages if document[i:i+len(x)] == x]
      if matches: break
      i += 1
      if i > len(document): raise Exception
    tok_to_char.append(i)
    i += len(matches[0])
  tok_to_char.append(i)  # After the last token
  return tok_to_char

def logits_to_probs(document, tokens, start_logits, end_logits, none_logit,
                    beam_size=None):
  span_logits = np.add.outer(start_logits, end_logits)
  tok_to_char = make_tok_to_char(document, tokens)
  all_logits = np.concatenate((np.array([none_logit]), span_logits.flatten()))
  log_partition = scipy.special.logsumexp(all_logits)
  all_probs_np = np.exp(all_logits - log_partition)
  all_probs = all_probs_np.tolist()  # for JSON serializability later
  p_na = all_probs[0]
  beam = []
  sorted_inds = np.argsort(-all_probs_np)
  for ind in sorted_inds:
    cur_prob = all_probs[ind]
    if beam_size and len(beam) == beam_size: break
    if ind == 0:
      beam.append(('', cur_prob))
    else:
      i, j = np.unravel_index(ind - 1, span_logits.shape)
      if i > j: continue
      start_char = tok_to_char[i]
      end_char = tok_to_char[j+1]
      phrase = document[start_char:end_char].strip()
      beam.append((phrase, cur_prob))
  return beam, p_na
