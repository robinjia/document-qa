<form action="/post_query" method="post" id="form">
<p><b>Enter a document:</b><br/>
<textarea rows="5" cols="100" name="document" id="d-area">
{{!document}}
</textarea>
</p>

<p><b>Enter a question:</b><br/>
<textarea rows="2" cols="100" name="question" id="q-area">
{{!question}}
</textarea>
</p>

<input type="submit" class="btn btn-primary"/>
</form>
<script type="text/javascript">
//<![CDATA[
var form = document.getElementById("form");
var d_area = document.getElementById("d-area");
var q_area = document.getElementById("q-area");
d_area.addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.keyCode == 13) {
    this.form.submit();
  }
});
q_area.addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.keyCode == 13) {
    this.form.submit();
  }
});
//]]>
</script>
