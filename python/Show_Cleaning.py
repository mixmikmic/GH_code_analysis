sc.addPyFile('../dist_workflow/shared.zip')

get_ipython().magic('run -i initilization.py')

from shared.Extension_to_timeit import pretty_time_result
from shared.GeneralDataImport import GeneralDataImport

dataIO = GeneralDataImport(
    parquet_path+"/normal_cluster_n_1000.parquet",
)

dataIO.select_features()

dataIO.select_id()

dataIO.select_labels()

print(dataIO.list_features)
print(dataIO.list_label)
print(dataIO.list_id)
df = dataIO.data_frame
df.limit(5).toPandas()
#df.limit(5).toPandas()

#ax = sb.regplot('a','b',df.toPandas(),fit_reg=False)
ax = sb.lmplot(
    'a',
    'b',
    dataIO._data_frame.toPandas(),
    fit_reg=False,
    size=8,
    hue='k',
    scatter_kws={'alpha':0.7,'s':60}
)
ax.ax.set_title('An initial look at data',fontsize=20)
plt.show()

#import data!
#from pyspark.sql import functions as F
#from shared.create_dummy_data import create_dummy_data 

#test_timer = %timeit -o feature_data = create_dummy_data(1000, "x y z", "label", outlier_number=0.2, outlier_factor=20)
#feature_data = feature_data.select([(10*F.col(i)).alias(i) for i in ["x","y","z"]])
#feature_data.orderBy('x',ascending=[0,0,0]).show()

# Select parameters
from cleaning.CreateParametersCleaning import ParamsCleaning

params = ParamsCleaning()
parameters = params.select_parameters()

parameters

from pyspark.ml import clustering

clustering.__all__

#print(params.output_parameters(parameters))
#test_params_1 = {'tol': 0.00001, 'k': 3, 'maxIter': 300, 'algorithm': 'GaussianMixture', 'seed': 1080866016001745000}
test_params_1 = params.output_parameters(parameters)
print(test_params_1)

from cleaning.ExecuteCleaningWorkflow import ExecuteWorkflow

partitions = [80]
sizes = [1000]
features = dataIO.list_features
labels = dataIO.list_label
#print(features)
#print(labels)

execution_model = ExecuteWorkflow(
    dict_params = test_params_1,
    cols_features = features, 
    cols_labels = labels)

# this is hardcoded at the moment, use the comment for testing purposes!
collection_of_data = [parquet_path+'/normal_cluster_n_{}.parquet'.format(i) for i in sizes]
collection_of_model = []
collection_of_transformed = []
#collection_of_data
#counts = [i.rdd.getNumPartitions() for i in collection_of_data]
#counts
#collection_of_data.append(df)

for jdx, partition_size in enumerate(partitions):
    for idx, data in enumerate(collection_of_data):

        df_data = (
            spark.
            read.
            parquet(data).
            repartition(partition_size)
        )
        
        iteration = idx+jdx*len(collection_of_data)
        logger_tester.info(
            'Iteration {} for data size {}'.
            format(iteration, sizes[idx])
        )

        model_timer = get_ipython().magic('timeit -r1 -o collection_of_model.append(execution_model.execute_pipeline(df_data)) ')
        transformer_timer = get_ipython().magic('timeit -o execution_model.apply_model(sc, collection_of_model[iteration], df_data)')
        collection_of_model = collection_of_model[:iteration+1]
        logger_tester.info(
            'Iteration '+str(iteration)+' for training model : '+pretty_time_result(model_timer))
        logger_tester.info(
            'Iteration '+str(iteration)+' for transforming model : '+pretty_time_result(transformer_timer))
        #merged_df.write.parquet('/home/svanhmic/workspace/data/DABAI/sparkdata/parquet/merged_df_parquet')

df_data = (
    spark.
    read.
    parquet(data).
    repartition(partition_size)
)

df_results = execution_model.apply_model(sc,
    collection_of_model[iteration],
    df_data
)
df_results.limit(5).toPandas()

execution_model.parameters

from cleaning.ShowCleaning import ShowResults
results = ShowResults(sc,
    execution_model.parameters,
    list_features=execution_model.features,
    list_labels=execution_model.labels)

prepared_df = results.prepare_table_data(df_results)

new_df = results.prepare_table_data(prepared_df, prediction_col='prediction')
summary_df = results.compute_summary(new_df)
summary_df.toPandas()

df_with_dists = results.select_cluster(new_df)

new_df.limit(5).toPandas()

from shared import Plot2DGraphs
from pyspark.sql import functions as F

if test_params_1['algorithm'] == 'GaussianMixture':
    Plot2DGraphs.plot_gaussians(
        new_df,
        execution_model.features,
        gaussian_std=2.5)
else:
    pdf_with_dists = new_df.toPandas()
    pdf_original = (dataIO.data_frame
                    .withColumn('k', F.col('k').cast('integer'))
                    .toPandas())
    
    pdf_with_dists.loc[:,'origin'] = 'kmeans'
    pdf_original.loc[:,'origin'] = 'original'
    pdf_original['prediction'] = pdf_original['k']
    pdf_merged = pd.concat([pdf_original, pdf_with_dists])
    g = sb.FacetGrid(pdf_merged,col="origin", hue="prediction",size=8)
    g.map(plt.scatter, "a", "b", alpha=.7)
    g.add_legend();
    g.set_titles(template='{col_name}')
    plt.show()   

df_with_dists = results.select_prototypes(df_results)

if test_params_1['algorithm'] == 'GaussianMixture':
    Plot2DGraphs.plot_gaussians(
        df_results,
        execution_model.features,
        gaussian_std=2)
else:
    pdf_with_dists = df_with_dists.toPandas()
    pdf_original = (dataIO.data_frame
                    .withColumn('k', F.col('k').cast('integer'))
                    .toPandas())
    
    pdf_with_dists.loc[:,'origin'] = 'kmeans'
    pdf_original.loc[:,'origin'] = 'original'
    pdf_original['prediction'] = pdf_original['k']
    pdf_merged = pd.concat([pdf_original, pdf_with_dists])
    g = sb.FacetGrid(pdf_merged,col="origin", hue="prediction",size=8)
    g.map(plt.scatter, "a", "b", alpha=.7)
    g.add_legend();
    g.set_titles(template='{col_name}')
    plt.show() 

