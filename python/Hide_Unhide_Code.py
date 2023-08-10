from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<strong>Click </strong><a href="javascript:code_toggle()"><strong>here</strong></a>.  To hide/unhide code cells.''')

print('Hide me!')

