from fgrad import embed

DC = '/Users/marcel/projects/HCP/dense_connectome/HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii'

DC_affinity = embed.preprocess_dense_connectome(DC)

embedding = embed.embed_dense_connectome(DC_affinity)

save_embedding(embedding, "../data/rsFC_eigenvectors.dscalar.nii")

