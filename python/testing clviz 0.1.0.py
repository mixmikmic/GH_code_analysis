import clarityviz as cl

# inImg = cl.get_raw_brain('Fear199')

# cl.run_pipeline('Fear199', 'RSA', resolution=5)

# get_regions(points_path, anno_path, output_path)
# rp = cl.get_regions('../Fear199_points.csv', '../img/Fear199_anno.nii', 'Fear199_regions.csv')

# create_graph(points_path, radius=20, output_filename=None)
g = cl.create_graph('../Fear199_regions.csv', 20, 'test.graphml')

figure = generate_region_graph('Fear199', 'Fear199_regions.csv', 'Fear199_region.html')

