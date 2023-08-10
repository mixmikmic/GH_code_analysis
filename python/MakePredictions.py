from src.data_pipeline import DataManager as DM
import src.model_builder as mb
import src.predict_one as p_1
import src.evaluator as ev
get_ipython().magic('matplotlib inline')

# # Create a new dataset
# DM(photo_dir= "path/to/dir/of/photos", dataset_name= 'Wildlife_ID_Data').process_photos()

# # Prepare Data: 
# Ftrs, Lbls, paths = mb.prep_data('DMtest', drop_hare=True)

# # Create a SVM
# svm = mb.create_SVC()
# model, X_test, y_test, y_pred, y_prob = mb.run_fit(svm, Ftrs, Lbls, test_size = .4)
# mb.save_model(model, 'my_svm')

# This function combines predict_one.predict() and evaluator.plot_probs()
# in a format convenient for jupyter notebooks
def pred_and_plot(photo):
    prediction = p_1.predict(photo, model = 'current_model')
    ev.plot_probs(prediction)
    return prediction

# result = pred_and_plot('Path/To/Photo.JPG')

result = pred_and_plot('images/misty_deer.JPG', )



