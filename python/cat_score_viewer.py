import score_tuner
import logging

scores_df, y_expected = score_tuner.load_pickled_scores("temp_pickled_scores_df.dump", "temp_pickled_y_expected.dump")

selected_category = "cdk"
category_scores_series = scores_df[selected_category]
category_scores_series.describe()

# small category plot
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

doc_number = range(max(category_scores_series.index))

correct_scores = category_scores_series[y_expected == selected_category]
incorrect_scores = category_scores_series[y_expected != selected_category]

category_beta = score_tuner.beta_for_categories_provider(y_expected)[selected_category]
opt_threshold = score_tuner.maximize_f_score(y_expected, category_scores_series, selected_category, category_beta)

plt.plot(incorrect_scores.index, incorrect_scores.values, 'yx')
plt.plot(correct_scores.index, correct_scores.values, 'gx')
plt.plot([min(category_scores_series.index), max(category_scores_series.index)], [opt_threshold]*2, 'r')

plt.axis([0, max(doc_number), 0, 1])
plt.show()

opt_threshold

selected_category = "openshift"
category_scores_series = scores_df[selected_category]
category_scores_series.describe()

# small category scores plot with threshold as based on f-score with selected beta for given category
import matplotlib.pyplot as plt

doc_number = range(max(category_scores_series.index))
correct_scores = category_scores_series[y_expected == selected_category]
incorrect_scores = category_scores_series[y_expected != selected_category]

category_beta = score_tuner.beta_for_categories_provider(y_expected)[selected_category]
opt_threshold = score_tuner.maximize_f_score(y_expected, category_scores_series, selected_category, category_beta)

plt.plot(incorrect_scores.index, incorrect_scores.values, 'yx')
plt.plot(correct_scores.index, correct_scores.values, 'gx')
plt.plot([min(category_scores_series.index), max(category_scores_series.index)], [opt_threshold]*2, 'r')

plt.axis([0, max(doc_number), 0, 1])
plt.show()

opt_threshold

score_df_norm = score_tuner.normalize_all_cats_scores(y_expected, scores_df)

selected_category = "cdk"
category_scores_series = score_df_norm[selected_category]
category_scores_series.describe()

# small category plot
import matplotlib.pyplot as plt
# %matplotlib inline

doc_number = range(max(category_scores_series.index))

correct_scores = category_scores_series[y_expected == selected_category]
incorrect_scores = category_scores_series[y_expected != selected_category]

category_beta = score_tuner.beta_for_categories_provider(y_expected)[selected_category]
opt_threshold = score_tuner.maximize_f_score(y_expected, category_scores_series, selected_category, category_beta)

plt.plot(incorrect_scores.index, incorrect_scores.values, 'yx')
plt.plot(correct_scores.index, correct_scores.values, 'gx')
plt.plot([min(category_scores_series.index), max(category_scores_series.index)], [opt_threshold]*2, 'r')

plt.axis([0, max(doc_number), 0, 1])
plt.show()

opt_threshold

selected_category = "openshift"
category_scores_series = score_df_norm[selected_category]
category_scores_series.describe()

# small category scores plot with threshold as based on f-score with selected beta for given category
import matplotlib.pyplot as plt

doc_number = range(max(category_scores_series.index))
correct_scores = category_scores_series[y_expected == selected_category]
incorrect_scores = category_scores_series[y_expected != selected_category]

category_beta = score_tuner.beta_for_categories_provider(y_expected)[selected_category]
opt_threshold = score_tuner.maximize_f_score(y_expected, category_scores_series, selected_category, category_beta)

plt.plot(incorrect_scores.index, incorrect_scores.values, 'yx')
plt.plot(correct_scores.index, correct_scores.values, 'gx')
plt.plot([min(category_scores_series.index), max(category_scores_series.index)], [opt_threshold]*2, 'r')

plt.axis([0, max(doc_number), 0, 1])
plt.show()

opt_threshold

score_tuner.f_score_for_category(y_expected, category_scores_series, "openshift", opt_threshold, category_beta)

score_tuner.f_score_for_category(y_expected, category_scores_series, "openshift", 0.5, category_beta)

