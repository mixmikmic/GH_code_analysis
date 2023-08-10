get_ipython().magic('run EUVIP_1_defaults.ipynb')

get_ipython().magic('pwd')

get_ipython().magic('cd -q ../test/')

mp = SparseEdges(parameter_file)
image = mp.imread(lena_file)
mp.pe.N = N
mp.pe.mask_exponent = 4.
mp.pe.line_width = 3.
mp.pe.figsize_edges = fig_width
mp.pe.figpath = '../docs/'
mp.init()
image = mp.normalize(image, center=False)
image *= mp.mask
print(image.min(), image.max())
fig, ax = mp.imshow(image, mask=True, norm=False)

name = experiment.replace('sparseness', 'lena')
matname = os.path.join(mp.pe.matpath, name + '.npy')
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=False)
    np.save(matname, edges)    

    
matname = os.path.join(mp.pe.matpath, name + '_rec.npy')
try:
    image_rec = np.load(matname)
except:
    image_rec = mp.reconstruct(edges, mask=True)        
    np.save(matname, image_rec)    
    

print(matname)

mp.pe

fig, a = mp.show_edges(edges, image=mp.dewhitening(image_rec), mask=True)#show_phase=False, 
mp.savefig(fig, name);

#list_of_number_of_edge =  np.logspace(0, 11, i, base=2)
#list_of_number_of_edge =  4**np.arange(6)
list_of_number_of_edge =  2* 4**np.arange(6)
print(list_of_number_of_edge)

fig, axs = plt.subplots(1, len(list_of_number_of_edge), figsize=(3*fig_width, 3*fig_width/len(list_of_number_of_edge)))
vmax = 1.
image_rec = mp.reconstruct(edges, mask=True)        
vmax = mp.dewhitening(image_rec).max()
for i_ax, number_of_edge in enumerate(list_of_number_of_edge):
    edges_ = edges[:, :number_of_edge][..., np.newaxis]
    image_rec = mp.dewhitening(mp.reconstruct(edges_, mask=True))
    fig, axs[i_ax] = mp.imshow(image_rec/vmax, fig=fig, ax=axs[i_ax], norm=False, mask=True)
    axs[i_ax].text(5, 40, 'N=%d' % number_of_edge, color='red', fontsize=21)
plt.tight_layout()
fig.subplots_adjust(hspace = .0, wspace = .0, left=0.0, bottom=0., right=1., top=1.)
mp.savefig(fig, name + '_movie');

vmax = 1.
image_rec = mp.reconstruct(edges, mask=True)        
vmax = mp.dewhitening(image_rec).max()
for i_ax, number_of_edge in enumerate(list_of_number_of_edge):
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width))
    edges_ = edges[:, :number_of_edge][..., np.newaxis]
    image_rec = mp.dewhitening(mp.reconstruct(edges_, mask=True))
    fig, ax = mp.imshow(image_rec/vmax, fig=fig, ax=ax, norm=False, mask=True)
    ax.text(5, 40, 'N=%d' % number_of_edge, color='red', fontsize=21)
    plt.tight_layout()
    fig.subplots_adjust(hspace = .0, wspace = .0, left=0.0, bottom=0., right=1., top=1.)

    mp.savefig(fig, name + '_movie_N' + str(number_of_edge), formats=['png']);

get_ipython().magic('cd -q ../notebooks/')

