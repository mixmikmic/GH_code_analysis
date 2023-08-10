get_ipython().run_line_magic('run', 'notebook_setup.ipynb')

cache = {}  # cache store for description string filtered to nouns only

temp_df = lith_code_desc.reset_index(drop=True)  # need numeric index

nrow, ncol = len(temp_df.index), len(temp_df.index)
token_sim = np.empty((nrow, ncol), dtype=np.float64)
exemplars = np.empty((nrow, ncol), dtype=np.float64)

# Custom similarity scorer
def filter_description(text, stopwords):
    tmp = text.split('-')[0]
    return get_nouns(tokenize_and_stem(tmp.strip(), stopwords))

for row in temp_df.itertuples():   
    if row.Index in cache:
        outer_stemmed = cache[row.Index]
    else:
        outer_stemmed = lemmatize_stems(filter_description(row.Description, stopwords))
    # End if

    for row2 in temp_df.itertuples():
        if row2.Index in cache:
            inner_stemmed = cache[row2.Index]
        else:
            inner_stemmed = lemmatize_stems(filter_description(row2.Description, stopwords))
            cache[row2.Index] = inner_stemmed
        # End if

        # Calculate similarity score
        score = calc_similarity_score(outer_stemmed, inner_stemmed)
        token_sim[row.Index, row2.Index] = score

        # Larger values for the given description increase chance that it will be selected as an exemplar
        # Doing this may or may not be useful...
        if len(outer_stemmed) < 3:
            exemplars[row.Index, row2.Index] = 1.0
        else:
            if len(outer_stemmed) > 5:
                exemplars[row.Index, row2.Index] = 0.0
            else:
                exemplars[row.Index, row2.Index] = 0.5
            # End if
        # End if
        
    # End for

# End for

print("If using amalgamated example dataset, this should be around 560 entries:", len(lith_code_desc.index))

plt.imshow(token_sim)
plt.colorbar()
plt.title('Similarity Matrix\n(1.0 is similar)');

from sklearn.manifold import MDS

# Cast to 2D space
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=5)
pos = mds.fit_transform(1 - token_sim)

plt.scatter(pos[:, 0], pos[:, 1], s=6);

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Convert to cosine similarity matrix
# , preference=-np.mean(exemplars)
# cos_sim = calc_similarity_score(pos)
ap_model = sklearn.cluster.AffinityPropagation(damping=0.8, affinity='precomputed')
ap_model.fit(token_sim)

cluster_centers_idx = ap_model.cluster_centers_indices_
labels = ap_model.labels_
unique_labels = np.unique(labels)
n_clusters_ = len(cluster_centers_idx)
print("Num clusters:", len(unique_labels))

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")

cluster_data = {}
for k in range(n_clusters_):
    class_members = labels == k
    
    cluster_size = len(class_members[class_members == True])
    exemplar = lith_desc[cluster_centers_idx[k]]
    exemplar = ' '.join(exemplar.split()[0:3])  # Get first three words in exemplar
    cluster_data[exemplar] = cluster_size
# End for

fig, ax = plt.subplots(figsize=(10,8))
clustered_info = pd.DataFrame.from_dict(cluster_data, orient='index')
clustered_info.index.name = 'Cluster'
clustered_info.columns = ['Members']
clustered_info.sort_index(ascending=False, inplace=True)
clustered_info.plot(kind='barh', fontsize=10, title=f'Number of Members\n(Estimated num clusters: {n_clusters_})', 
                    legend=False, ax=ax);

# Write out results to csv
res = {}

for cluster_id in unique_labels:
    exemplar = lith_desc[cluster_centers_idx[cluster_id]]
    cluster = np.unique(lith_desc[np.nonzero(labels == cluster_id)[0]])
    res[exemplar] = " | ".join(cluster)
# End for

res_df = pd.DataFrame([i for i in res.values()], index=res.keys(), columns=["Matches"])
res_df.index.name = "Exemplar"

try:
    res_df.to_csv("ap_output.csv", index=True)
except PermissionError:
    raise PermissionError("ERROR OCCURRED - csv file is probably open in Excel. Close Excel and try again.")
# End try

assert len(unique_labels) == len(res_df.index), "Number of clusters do not match outputted cluster exemplars!"

fig, ax = plt.subplots(figsize=(12,10))

cm = plt.get_cmap('Set1')
colors = [cm(1.0*i/n_clusters_) for i in range(n_clusters_)]

count = 0
# for each cluster get all members and plot with same color
for cluster_idx, color in zip(range(n_clusters_), colors):
    in_cluster = labels == cluster_idx
    cluster_set = pos[in_cluster]
    
    cluster_row = cluster_centers_idx[cluster_idx]
    cluster_center = pos[cluster_row]
    
    exemplar = lith_desc[cluster_row]
    
    # Plot cluster exemplar
    ax.scatter(cluster_center[0], cluster_center[1], marker='o', c=color, s=100)
    ax.annotate(" ".join(exemplar.split()[:5]), (cluster_center[0], cluster_center[1]), 
                alpha=1, fontsize=12)
    
    for i, txt in enumerate(lith_desc[in_cluster]):
        if txt == exemplar:
            continue
            
        x, y = cluster_set[i]
        
        # Plot cluster member
        ax.scatter(x, y, marker='o', c=color, alpha=0.5, s=10)
        
        # Plot line between cluster exemplar and member
        plt.plot([cluster_center[0], x], [cluster_center[1], y], c=color, 
                 linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Add text annotation
        # ax.annotate(" ".join(txt.split()[0:3]), (x, y), alpha=0.1)
    # End for
    
# End for

plt.title(f"Estimated number of clusters: {n_clusters_}");



