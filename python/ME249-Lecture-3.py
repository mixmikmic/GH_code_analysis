get_ipython().magic('matplotlib inline')
# plots graphs within the notebook
get_ipython().magic("config InlineBackend.figure_format='svg' # not sure what this does, may be default images to svg format")

from IPython.display import Image

from IPython.core.display import HTML
def header(text):
    raw_html = '<h4>' + str(text) + '</h4>'
    return raw_html

def box(text):
    raw_html = '<div style="border:1px dotted black;padding:2em;">'+str(text)+'</div>'
    return HTML(raw_html)

def nobox(text):
    raw_html = '<p>'+str(text)+'</p>'
    return HTML(raw_html)

def addContent(raw_html):
    global htmlContent
    htmlContent += raw_html
    
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
    
font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 18,
        }


import matplotlib.pyplot as plt
import numpy as np

Lx = 2.*np.pi
Nx = 64
u = np.zeros(Nx,dtype='float64')
du = np.zeros(Nx,dtype='float64')
ddu = np.zeros(Nx,dtype='float64')
k_0 = 2.*np.pi/Lx 
x = np.linspace(Lx/Nx,Lx,Nx)
Nwave = 24
uwave = np.zeros((Nx,Nwave),dtype='float64')
duwave = np.zeros((Nx,Nwave),dtype='float64')
dduwave = np.zeros((Nx,Nwave),dtype='float64')
ampwave = np.random.random(Nwave)
phasewave = np.random.random(Nwave)*2*np.pi
for iwave in range(Nwave):
    uwave[:,iwave] = ampwave[iwave]*np.cos(k_0*iwave*x+phasewave[iwave])
    duwave[:,iwave] = -k_0*iwave*ampwave[iwave]*np.sin(k_0*iwave*x+phasewave[iwave])
    dduwave[:,iwave] = -(k_0*iwave)**2*ampwave[iwave]*np.cos(k_0*iwave*x+phasewave[iwave])
u = np.sum(uwave,axis=1)
du = np.sum(duwave,axis=1)
ddu = np.sum(dduwave,axis=1)
#print(u)
plt.plot(x,u,lw=2)
plt.xlim(0,Lx)
#plt.legend(loc=3, bbox_to_anchor=[0, 1],
#           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$x$', fontdict = font)
plt.ylabel('$u$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
plt.plot(x,du,lw=2)
plt.xlim(0,Lx)
#plt.legend(loc=3, bbox_to_anchor=[0, 1],
#           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$x$', fontdict = font)
plt.ylabel('$du/dx$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
plt.plot(x,ddu,lw=2)
plt.xlim(0,Lx)
#plt.legend(loc=3, bbox_to_anchor=[0, 1],
#           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$x$', fontdict = font)
plt.ylabel('$d^2u/dx^2$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

#check FT^-1(FT(u)) - Sanity check
u_hat = np.fft.fft(u)
v = np.real(np.fft.ifft(u_hat))

plt.plot(x,u,'r-',lw=2,label='$u$')
plt.plot(x,v,'b--',lw=2,label='$FT^{-1}[FT[u]]$')
plt.xlim(0,Lx)
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$x$', fontdict = font)
plt.ylabel('$u$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
print('error',np.linalg.norm(u-v,np.inf))

F = np.zeros(Nx/2+1,dtype='float64')
F = np.real(u_hat[0:Nx/2+1]*np.conj(u_hat[0:Nx/2+1]))
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0))) #This is how the FFT stores the wavenumbers
plt.loglog(k[0:Nx/2+1],F,'r-',lw=2,label='$\Phi_u$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$k$', fontdict = font)
plt.ylabel('$\Phi_u(k)$', fontdict = font)
plt.xticks(fontsize = 16)
y_ticks = np.logspace(-30,5,8)
plt.yticks(y_ticks,fontsize = 16) #Specify ticks, necessary when increasing font
plt.show()

# filtering the smaller waves
def low_pass_filter_fourier(a,k,kcutoff):
    N = a.shape[0]
    a_hat = np.fft.fft(a)
    filter_mask = np.where(np.abs(k) > kcut)
    a_hat[filter_mask] = 0.0 + 0.0j
    a_filter = np.real(np.fft.ifft(a_hat))
    return a_filter
kcut=Nwave/2+1
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
v = low_pass_filter_fourier(u,k,kcut)
u_filter_exact = np.sum(uwave[:,0:kcut+1],axis=1)
plt.plot(x,v,'r-',lw=2,label='filtered with fft')
plt.plot(x,u_filter_exact,'b--',lw=2,label='filtered (exact)')
plt.plot(x,u,'g:',lw=2,label='original')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$x$', fontdict = font)
plt.ylabel('$u$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
print('error:',np.linalg.norm(v-u_filter_exact,np.inf))

F = np.zeros(Nx/2+1,dtype='float64')
F_filter = np.zeros(Nx/2+1,dtype='float64')
u_hat = np.fft.fft(u)
F = np.real(u_hat[0:Nx/2+1]*np.conj(u_hat[0:Nx/2+1]))
v_hat = np.fft.fft(v)
F_filter = np.real(v_hat[0:Nx/2+1]*np.conj(v_hat[0:Nx/2+1]))
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
plt.loglog(k[0:Nx/2+1],F,'r-',lw=2,label='$\Phi_u$')
plt.loglog(k[0:Nx/2+1],F_filter,'b-',lw=2,label='$\Phi_v$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$k$', fontdict = font)
plt.ylabel('$\Phi_u(k)$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(y_ticks,fontsize = 16)
plt.show()

u_hat = np.fft.fft(u)
kfilter = Nwave/2
k = np.linspace(0,Nx-1,Nx)
filter_mask = np.where((k < kfilter) | (k > Nx-kfilter) )
u_hat[filter_mask] = 0.+0.j
v = np.real(np.fft.ifft(u_hat))
plt.plot(x,v,'r-',lw=2,label='Filtered (FT)')
plt.plot(x,np.sum(uwave[:,kfilter:Nwave+1],axis=1),'b--',lw=2,label='Filtered (exact)')
plt.plot(x,u,'g:',lw=2,label='original')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0,Lx)
plt.show()

F = np.zeros(Nx/2+1,dtype='float64')
F_filter = np.zeros(Nx/2+1,dtype='float64')
u_hat = np.fft.fft(u)
F = np.real(u_hat[0:Nx/2+1]*np.conj(u_hat[0:Nx/2+1]))
v_hat = np.fft.fft(v)
F_filter = np.real(v_hat[0:Nx/2+1]*np.conj(v_hat[0:Nx/2+1]))
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
plt.loglog(k[0:Nx/2+1],F,'r-',lw=2,label=r'$\Phi_u$')
plt.loglog(k[0:Nx/2+1],F_filter,'b-',lw=2,label=r'$\Phi_v$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$k$', fontdict = font)
plt.ylabel('$\Phi(k)$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(y_ticks,fontsize = 16)
plt.show()

u_hat = np.fft.fft(u)
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
ik = 1j*k
v_hat = ik*u_hat

v = np.real(np.fft.ifft(v_hat))
plt.plot(x,v,'r-',lw=2,label='$FT^{-1}[\hat{\jmath}kFT[u]]$')
plt.plot(x,du,'b--',lw=2,label='exact derivative')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.ylabel('$du/dx$',fontsize = 18)
plt.xlabel('$x$',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0,Lx)
plt.show()
print('error:',np.linalg.norm(v-du))

mk2 = ik*ik
v_hat = mk2*u_hat

v = np.real(np.fft.ifft(v_hat))
plt.plot(x,v,'r-',lw=2,label='$FT^{-1}[-k^2FT[u]]$')
plt.plot(x,ddu,'b--',lw=2,label='exact derivative')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.ylabel('$d^2u/dx^2$',fontsize = 18)
plt.xlabel('$x$',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0,Lx)
plt.show()
print('error:',np.linalg.norm(v-ddu))

du_fd = np.zeros(Nx,dtype='float64')
dx = Lx/Nx
du_fd[1:Nx] = (u[1:Nx]-u[0:Nx-1])/dx
du_fd[0] = (u[0]-u[Nx-1])/dx
plt.plot(x,du_fd,'r-',lw=2,label='FD derivative')
plt.plot(x,du,'b--',lw=2,label='exact derivative')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.ylabel('$du/dx$',fontsize = 18)
plt.xlabel('$x$',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0,Lx)
plt.show()
print('error:',np.linalg.norm(du_fd-du,np.inf))

F = np.zeros(Nx/2+1,dtype='float64')
F_fd = np.zeros(Nx/2+1,dtype='float64')
du_hat = np.fft.fft(du)/Nx
F = np.real(du_hat[0:Nx/2+1]*np.conj(du_hat[0:Nx/2+1]))
dv_hat = np.fft.fft(du_fd)/Nx
F_fd = np.real(dv_hat[0:Nx/2+1]*np.conj(dv_hat[0:Nx/2+1]))
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
plt.loglog(k[0:Nx/2+1],F,'r-',lw=2,label='$\Phi_{du/dx}$')
plt.loglog(k[0:Nx/2+1],F_fd,'b-',lw=2,label='$\Phi_{\delta u/\delta x}$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$k$', fontdict = font)
plt.ylabel('$\Phi(k)$', fontdict = font)
plt.xticks(fontsize = 16)
plt.ylim(1e-3,250)
plt.yticks(fontsize = 16)
plt.show()
print('error:',np.linalg.norm(F[0:Nwave/2]-F_fd[0:Nwave/2],np.inf))
print('error per wavenumber')
plt.loglog(k[0:Nx/2+1],np.abs(F-F_fd),'r-',lw=2)
plt.xlabel('$k$', fontdict = font)
plt.ylabel('$\Vert\Phi_{d u/d x}(k)-\Phi_{\delta u/\delta x}(k)\Vert$', fontdict = font)
plt.xticks(fontsize = 16)
plt.ylim(1e-5,250)
plt.yticks(fontsize = 16)
plt.show()

L = np.pi
N = 32
dx = L/N
omega = np.linspace(0,L,N)
omegap_exact = omega
omegap_modified = np.sin(omega)
plt.plot(omega,omegap_exact,'k-',lw=2,label='exact')
plt.plot(omega,omegap_modified,'r-',lw=2,label='1st order upwind')
plt.xlim(0,L)
plt.ylim(0,L)
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.xlabel('$\omega\Delta x$', fontdict = font)
plt.ylabel('$\omega_{mod}\Delta x$', fontdict = font)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()

u_hat = np.fft.fft(u)
k = np.hstack((np.arange(0,Nx/2+1),np.arange(-Nx/2+1,0)))
dx = Lx/Nx
ikm = 1j*np.sin(2.*np.pi/Nx*k)/dx+(1-np.cos(2.*np.pi/Nx*k))/dx
v_hat = ikm*u_hat
dum = np.real(np.fft.ifft(v_hat))
plt.plot(x,dum,'r-',lw=2,label='$FT^{-1}[\hat{\jmath}\omega_{mod}FT[u]]$')
plt.plot(x,du_fd,'b--',lw=2,label='$\delta u/\delta x$')
plt.legend(loc=3, bbox_to_anchor=[0, 1],
           ncol=3, shadow=True, fancybox=True)
plt.ylabel('$du/dx$',fontsize = 18)
plt.xlabel('$x$',fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim(0,Lx)
plt.show()
print('error:',np.linalg.norm(du_fd-dum,np.inf))

get_ipython().system('ipython nbconvert --to html ME249-Lecture-3-YOURNAME.ipynb')

