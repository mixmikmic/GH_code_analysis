from IPython.display import HTML

HTML('''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>
<ul>
<li>The code is in Python. Matlab does not work with the interactive elements used in this notebook.
<li>The current version of ipywidgets shows only static images online.  Interactivity can only be viewed by downloading and running in a Jupyter Notebook server.  The next version of ipywidgets will permit online viewing of interactivity.
</ul>
''')

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

x = np.linspace(0, 4.9999, 1000)

plt.figure()
plt.plot(x, -np.log(5 - x))
plt.xlim([0, 5.2])
plt.show()

from ipywidgets import interact

@interact(mu=(0.1, 2.0, 0.1))
def logbarrier(mu):
    f = -mu*np.log(5 - x)
    plt.plot(x, f)
    plt.xlim([0, 5.1])
    plt.ylim([-2, 2])
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()

plt.figure()
plt.plot(x, -x, 'k--')
mu = 1.0
f = -x - mu*np.log(5 - x)
plt.plot(x, f)
plt.xlim([0, 5.1])
plt.ylim([-5, 2])
plt.xlabel('x')
plt.ylabel('f')
plt.show()

@interact(mu=(0.1, 2.0, 0.1))
def barrier(mu):
    f = -x - mu*np.log(5 - x)
    plt.plot(x, -x, 'k--')
    plt.plot(x, f)
    plt.xlim([0, 5.1])
    plt.ylim([-5, 2])
    plt.xlabel('x')
    plt.ylabel('f')
    idx = np.argmin(f)
    plt.plot(x[idx], f[idx], 'r*', ms=14)
    plt.show()

plt.figure()
plt.plot(x, -x, 'k--')

muvec = [1.0, 0.5, 0.25, 0.125]

for mu in muvec:

    f = -x - mu*np.log(5 - x)
    plt.plot(x, f)

plt.xlim([0, 5.1])
plt.ylim([-5, 2])
plt.xlabel('x')
plt.ylabel('f')
plt.show()



