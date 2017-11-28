% rebase('main.tpl')
<h2>DocumentQA, No-Answer variant (Single Model)</h2>
<p><b>Document:</b> {{document}}</p>
<p><b>Question:</b> {{question}}</p>
<p><b>Model Predictions:</b>
<ol>
% for phrase, prob in beam:
  % if not phrase:
    % phrase = '<No Answer>'
  % end
  <li> [p={{'%.3g' % prob}}] {{phrase}}
% end
</ol>
</p>
<p><b>No-Answer Probability:</b> {{'%.5f' % p_na}}</p>
% include('form.tpl', document=document, question=question)
