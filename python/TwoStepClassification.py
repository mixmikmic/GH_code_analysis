import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
import process as p
import dmc

df = p.processed_data()
for c in [col for col in df.columns if 'Prob' in col]:
    df = df.drop(c, 1)

def predict_return_quantity_direct(df, tr_size, te_size):
    results = []
    X, Y = dmc.transformation.transform(df, scaler=dmc.normalization.scale_features,
                                        binary_target=False)
    train = X[:tr_size], Y[:tr_size]
    test = X[tr_size:tr_size + te_size], Y[tr_size:tr_size + te_size]
    for classifier in p.basic[:-1]:
        clf = classifier(train[0], train[1])
        res = clf(test[0])
        precision = dmc.evaluation.precision(res, test[1])
        cost = dmc.evaluation.dmc_cost(res, test[1])
        results.append((precision, cost))
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])

def predict_return_quantity_twostep(df, tr_size, te_size):
    results = []
    X, Y = dmc.transformation.transform(df, scaler=dmc.normalization.scale_features,
                                        binary_target=True)
    Y_fin = dmc.transformation.transform_target_vector(df, binary=False)
    train = X[:tr_size], Y[:tr_size]
    test = X[tr_size:tr_size + te_size], Y[tr_size:tr_size + te_size]
    for classifier in p.basic[:-1]:
        clf = classifier(train[0], train[1])
        res = clf(test[0])
        Y_csr, res_csr = csr_matrix(Y).T, csr_matrix(res).T
        train_fin = hstack([train[0], Y_csr[:tr_size]]), Y_fin[:tr_size]
        test_fin = hstack([test[0], res_csr]), Y_fin[tr_size:tr_size + te_size]
        clf_fin = classifier(train_fin[0], train_fin[1])
        res_fin = clf_fin(test_fin[0])
        precision = dmc.evaluation.precision(res_fin, test_fin[1])
        cost = dmc.evaluation.dmc_cost(res_fin, test_fin[1])
        results.append((precision, cost))
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])

def benchmark_prediction_target(df, tr_size, te_size, samplings=10):
    df_res = pd.DataFrame(index=p.basic[:-1])
    for i in range(samplings):
        df = p.shuffle(df)
        dfc = df[:te_size + tr_size].copy()
        res_dir = predict_return_quantity_direct(dfc, tr_size, te_size)
        res_two = predict_return_quantity_twostep(dfc, tr_size, te_size)
        df_res[str(i) + '_precision'] = res_two[0] - res_dir[0]
        df_res[str(i) + '_cost'] = res_dir[1] - res_two[1]
    return df_res

benchmark_prediction_target(df, 4000, 20000, 5)



