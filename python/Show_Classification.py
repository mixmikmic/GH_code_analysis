get_ipython().magic('run -i initilization.py')

from classification.ExecuteClassificationWorkflow import ExecuteWorkflowClassification
import classification.CreateParametersClasification as create_params
from shared import GeneralDataImport
from IPython.display import display

data_import = GeneralDataImport.GeneralDataImport(parquet_path+'/merged_cvr.parquet')

data_import.select_columns()

from pyspark.sql import functions as F

train_df, test_df = (data_import
                     .data_frame
                     .filter(F.col('label') < 2.0)
                     .randomSplit([0.66, 0.33])
                    )

#print(data_import.list_features)
print('Number of training points are {}'.format(train_df.count()))
print('Number of test points are {}'.format(test_df.count()))
train_df.limit(5).toPandas()
#train_df.printSchema()

selector = create_params.ParamsClassification()
params = selector.select_parameters()
display(params)

parameter_dict = selector.output_parameters(params)
parameter_dict

model = ExecuteWorkflowClassification(
    parameter_dict,
    data_import.standardize,
    data_import.list_features
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
result_model = model.pipeline.fit(train_df)
crossfitted_model = model.run_cross_val(
    train_df,
    BinaryClassificationEvaluator(),
    3
)
#summary = fitted_data.bestModel.stages[-1].summary

df_no_cv_pipeline = (result_model.transform(test_df))
l = model.pipeline.getStages()[-1].getLabelCol()
p = model.pipeline.getStages()[-1].getPredictionCol()
df_confusion = df_no_cv_pipeline.groupBy([l,p]).count()
df_confusion.toPandas()

if crossfitted_model.bestModel.stages[-1].hasSummary:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20, 14))
    summary = crossfitted_model.bestModel.stages[-1].summary
    
    
    print('The area under the curve is {}'.format(summary.areaUnderROC))
    attributes = []
    titles = ['F-measure by Threshold','Precision by Recall','Precision by Threshold', 'ROC', 'Recall by Threshold']
    attributes.append(summary.fMeasureByThreshold.toPandas())
    attributes.append(summary.pr.toPandas())
    attributes.append(summary.precisionByThreshold.toPandas())
    attributes.append(summary.roc.toPandas())
    attributes.append(summary.recallByThreshold.toPandas())
    #iterations = summary.totalIterations
    
    jdx = 0
    for idx, data_frame in enumerate(attributes):
        if idx % 3 == 0 and idx != 0:
            jdx+=1
        ax = axes[jdx,idx % 3]
        ax.plot(data_frame.columns[0],
                data_frame.columns[1],
                data=data_frame,                
               )
        ax.legend()
        ax.set_xlabel(data_frame.columns[0])
        ax.set_ylabel(data_frame.columns[1])
        ax.set_title(titles[idx])
    plt.show()

from classification import ShowClassification

show_classification_attributes(crossfitted_model)



