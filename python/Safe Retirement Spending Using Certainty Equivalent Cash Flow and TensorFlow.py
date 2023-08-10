from __future__ import print_function

import argparse
import pickle
from time import strftime
import sys
import six
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import lifetable

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import matplotlib.mlab as mlab
import plotly.plotly as py


# TensorFlow numeric type to use for floating point variables
# tf.float32 is 2x faster but less accurate
# tf.float64 will run out of accuracy for high gamma (> 8)
float_type = tf.float64

############################################################
# returns 1928-2015
############################################################

first_year = 1928
last_year = 2015
years = range(first_year, last_year+1) # pythonically yields [1928, 1929...2015]
years_history = len(years)
years_retired = 30
num_cohorts = years_history - years_retired + 1
num_assets = 2

bestfile = "jupyter"
#gamma = 1.0

sp500 = pd.Series([
    0.4381,-0.083,-0.2512,-0.4384,-0.0864,0.4998,-0.0119,0.4674,0.3194,-0.3534,0.2928,-0.011,
    -0.1067,-0.1277,0.1917,0.2506,0.1903,0.3582,-0.0843,0.052,0.057,0.183,0.3081,0.2368,0.1815,
    -0.0121,0.5256,0.326,0.0744,-0.1046,0.4372,0.1206,0.0034,0.2664,-0.0881,0.2261,0.1642,0.124,
    -0.0997,0.238,0.1081,-0.0824,0.0356,0.1422,0.1876,-0.1431,-0.259,0.37,0.2383,-0.0698,0.0651,
    0.1852,0.3174,-0.047,0.2042,0.2234,0.0615,0.3124,0.1849,0.0581,0.1654,0.3148,-0.0306,0.3023,
    0.0749,0.0997,0.0133,0.372,0.2268,0.331,0.2834,0.2089,-0.0903,-0.1185,-0.2197,0.2836,0.1074,
    0.0483,0.1561,0.0548,-0.3655,0.2594,0.1482,0.021,0.1589,0.3215,0.1352,0.0136],
    index = years)

bonds=pd.Series([
    0.0084,0.042,0.0454,-0.0256,0.0879,0.0186,0.0796,0.0447,0.0502,0.0138,0.0421,0.0441,
    0.054,-0.0202,0.0229,0.0249,0.0258,0.038,0.0313,0.0092,0.0195,0.0466,0.0043,-0.003,
    0.0227,0.0414,0.0329,-0.0134,-0.0226,0.068,-0.021,-0.0265,0.1164,0.0206,0.0569,0.0168,
    0.0373,0.0072,0.0291,-0.0158,0.0327,-0.0501,0.1675,0.0979,0.0282,0.0366,0.0199,0.0361,
    0.1598,0.0129,-0.0078,0.0067,-0.0299,0.082,0.3281,0.032,0.1373,0.2571,0.2428,-0.0496,
    0.0822,0.1769,0.0624,0.15,0.0936,0.1421,-0.0804,0.2348,0.0143,0.0994,0.1492,-0.0825,
    0.1666,0.0557,0.1512,0.0038,0.0449,0.0287,0.0196,0.1021,0.201,-0.1112,0.0846,0.1604,
    0.0297,-0.091,0.1075,0.0128],
                index=years)

cpi=pd.Series([
    -0.0115607,0.005848,-0.0639535,-0.0931677,-0.1027397,0.0076336,0.0151515,0.0298507,
    0.0144928,0.0285714,-0.0277778,0,0.0071429,0.0992908,0.0903226,0.0295858,0.0229885,
    0.0224719,0.1813187,0.0883721,0.0299145,-0.0207469,0.059322,0.06,0.0075472,0.0074906,
    -0.0074349,0.0037453,0.0298507,0.0289855,0.0176056,0.017301,0.0136054,0.0067114,0.0133333,
    0.0164474,0.0097087,0.0192308,0.0345912,0.0303951,0.0471976,0.0619718,0.0557029,0.0326633,
    0.0340633,0.0870588,0.1233766,0.0693642,0.0486486,0.0670103,0.0901771,0.1329394,0.125163,
    0.0892236,0.0382979,0.0379098,0.0394867,0.0379867,0.010979,0.0443439,0.0441941,0.046473,
    0.0610626,0.0306428,0.0290065,0.0274841,0.026749,0.0253841,0.0332248,0.017024,0.016119,
    0.0268456,0.0338681,0.0155172,0.0237691,0.0187949,0.0325556,0.0341566,0.0254065,0.0408127,
    0.0009141,0.0272133,0.0149572,0.0296,0.0174,0.015,0.0076,0.0073],
              index=years)

real_stocks = sp500 - cpi
real_bonds = bonds - cpi

startval = 100
years_retired = 30
# 1% constant spending
const_spend_pct = .0225
const_spend = startval * const_spend_pct

# var spending a function of years left
var_spend_pcts = pd.Series([ 0.5/(30-ix) for ix in range(30)])
var_spend_pcts[29] = 1.0

# stocks starting at 82%, decreasing 0.5% per year
stock_allocations = pd.Series([0.82 - 0.005* ix for ix in range(30)])
bond_allocations = 1 - stock_allocations

pickle_list = [const_spend, var_spend_pcts, stock_allocations, bond_allocations]
pickle.dump( pickle_list, open( bestfile, "wb" ) )



class SafeWithdrawalModel:
    """initialize graph and parameters shared by all retirement cohorts"""
    def __init__(self,
                 returns_list, # series with returns for assets
                 names_list, # names of assets
                 allocations_list, # list of % allocated to each asset class
                 start_val, # starting portfolio value e.g. 100
                 const_spend_pct,
                 var_spend_pcts,
                 gamma,
                 survival,
                 verbose=False):

        # read params, initialize Tensorflow graph and session
        # set up ops specific to model
        self.verbose=verbose
        self.startval=startval
        self.returns_list = returns_list
        self.names_list = names_list
        self.num_assets = len(self.names_list)
        self.start_val = start_val
        self.ret_years = len(allocations_list[0])
        self.const_spend_pct = const_spend_pct
        self.var_spend_pcts = var_spend_pcts
        self.survival=survival
        self.gamma = gamma

        # model will have a cohort_history object, optimizer object
        # initialize with placeholder, needs rest of model initialized first
        self.cohort_history = None
        self.optimizer = None

        self.first_year = returns_list[0].index[0]
        self.last_year = returns_list[0].index[-1]
        self.total_cohorts = len(returns_list[0])
        self.ret_cohorts = self.total_cohorts - self.ret_years + 1

        print('%s Create TensorFlow graph and session' % strftime("%H:%M:%S"))
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        self.return_ops = []
        self.allocation_ops = []

        with self.graph.as_default():
            with tf.device("/cpu:0"):
                # some constants
                self.zero = tf.constant(0.0, dtype=float_type, name="zero")
                self.one = tf.constant(1.0, dtype=float_type, name="one")
                self.one_hundred = tf.constant(100.0, dtype=float_type, name="one_hundred")
                self.ten_thousand = tf.constant(10000.0, dtype=float_type, name="ten_thousand")
                self.one_hundred_thousand = tf.constant(100000.0, dtype=float_type, name="one_million")
                self.one_million = tf.constant(1000000.0, dtype=float_type, name="one_million")
                self.very_small_amts = tf.constant(np.array([0.000001] * self.ret_years),
                                                   dtype=float_type, name="very_small_amts")
                self.zero_years = tf.constant(np.zeros(self.ret_years),
                                              dtype=float_type, name = "zero_years")
                self.one_years = tf.constant(np.ones(self.ret_years), dtype=float_type, name="one_years")
                self.ret_years_op = tf.constant(self.ret_years, dtype=float_type, name="ret_years")
                #gamma
                self.gamma_op = tf.constant(gamma, dtype=float_type, name="gamma")
                self.one_minus_gamma = tf.sub(self.one, self.gamma, name="one_minus_gamma")
                self.inv_one_minus_gamma = tf.div(self.one, self.one_minus_gamma,
                                                  name="inv_one_minus_gamma")

                self.cost_multiplier = self.ten_thousand

                # generate op for start_val
                self.start_val_op = tf.constant(100.0, dtype=float_type, name ="port_start_val")

                # generate ops for returns
                for prefix, return_series in zip(names_list, returns_list):
                    self.return_ops.append(self.gen_tf_const_list(return_series, "%s_return" % prefix,
                                                                  verbose=self.verbose))

                # only implemented for n=2 assets
                # generate ops for allocations for first n-1 assets
                prefix = names_list[0]
                alloc_series = allocations_list[0]
                stock_alloc_ops = self.gen_tf_var_list(alloc_series, "%s_alloc" % prefix,
                                                       verbose=self.verbose)
                self.allocation_ops.append(stock_alloc_ops)

                # ops for soft constraints: 0 < stock allocation < 1
                self.alloc_min_0_ops = self.gen_zero_min_list(stock_alloc_ops, "alloc_min_0",
                                                              verbose=self.verbose)
                self.cost_alloc_min_0_op = tf.mul(self.cost_multiplier,
                                                  tf.add_n(self.alloc_min_0_ops,
                                                           name="cost_alloc_min_0"),
                                                  name="cost_alloc_min_0_mult")

                self.alloc_max_1_ops = self.gen_one_max_list(stock_alloc_ops, "alloc_max_1",
                                                             verbose=self.verbose)
                self.cost_alloc_max_1_op = tf.mul(self.cost_multiplier,
                                                  tf.add_n(self.alloc_max_1_ops, name = "cost_alloc_max_1"))

                # ops for soft constraints: declining stock allocation
                # why? for example, 1966 is the worst cohort, and 1974 is its worst stock return (-40%)
                # to maximize CE, optimization sets stock allocation at a minimum to not run out of money
                # in worst year. It will go e.g. 80% stock alloc in year 8 and 56% in year 9, return to
                # 80% in year 10.To avoid artifacts like that, knowing stock allocation should decline
                # over time, we add a large penalty to objective when stock allocation increases
                # from one year to next.
                self.alloc_decrease_ops = self.gen_diff_list(stock_alloc_ops, "alloc_decrease",
                                                             verbose=self.verbose)
                self.cost_alloc_decrease_op = tf.mul(self.cost_multiplier,
                                                     tf.add_n(self.alloc_decrease_ops,
                                                              name="alloc_decrease_cost_op"))
                # last asset is 1-previous assets
                bond_alloc_ops = []

                var_prefix = "%s_alloc" % names_list[1]
                print ('%s Create ops for %s' % (strftime("%H:%M:%S"), var_prefix))
                for ix, op in enumerate(stock_alloc_ops):
                    var_name = "%s_%d" % (var_prefix, ix)
                    if self.verbose:
                        print('Create %s' % var_name)
                    var_op = tf.sub(self.one, stock_alloc_ops[ix], name=var_name)
                    bond_alloc_ops.append(var_op)
                self.allocation_ops.append(bond_alloc_ops)

                # generate ops for const, var spending
                self.const_spend_pct_op = tf.Variable(const_spend_pct, dtype=float_type, name="const_spend_pct")
                self.sess.run(self.const_spend_pct_op.initializer)
                self.const_spending_op = tf.mul(self.const_spend_pct_op, self.one_hundred, name="const_spend")

                self.var_spending_ops = self.gen_tf_var_list(self.var_spend_pcts, "var_spend",
                                                             verbose=self.verbose)

                # all ops to be trained
                self.all_var_ops = [self.const_spend_pct_op] +                                    self.var_spending_ops +                                    self.allocation_ops[0]

                # op for soft constraint: const spending > 0
                self.cspend_min_0_op = tf.maximum(self.zero, tf.neg(self.const_spend_pct_op,
                                                                    name="neg_cspend_min_0_op"),
                                                  name="cspend_min_0_op")
                self.cost_cspend_min_0_op = tf.mul(self.cost_multiplier,
                                                   self.cspend_min_0_op,
                                                   name="cost_cspend_min_0")

                # op for soft constraint: var spending > 0
                self.vspend_min_0_ops = self.gen_zero_min_list(self.var_spending_ops, "vspend_min_0",
                                                               verbose=self.verbose)
                self.cost_vspend_min_0_op = tf.mul(self.cost_multiplier,
                                                   tf.add_n(self.vspend_min_0_ops,
                                                            name="cost_vspend_min_0"))

                if survival is not None:
                    survival_array=np.array(survival)
                    self.survival_tensor = tf.constant(survival_array, dtype=float_type,
                                                       name="survival_tensor")

                # global step counter
                self.step_count = tf.Variable(0, dtype=float_type, name="step_count", trainable=False)
                self.increment_step = self.step_count.assign_add(1)

                #init op
                self.init_op = tf.initialize_all_variables()

    def __del__(self):
        """When deleting model, close session, clear default graph"""
        print("Destructor reset graph")
        try:
            with self.graph.as_default():
                tf.reset_default_graph()
        except Exception, e:
            print ("Destructor couldn't reset graph: %s" % str(e))

        try:
            print ("Destructor close Tensorflow session")
            self.sess.close()
        except Exception, e:
            print ("Destructor couldn't close session: %s" % str(e))


    def gen_tf_const_list(self, const_iter, const_prefix, start_index=0, verbose=False):
        """take a list or iterator of values, generate and return tensorflow constant ops for each"""
        print ('%s Create constants %s' % (strftime("%H:%M:%S"), const_prefix))
        with self.graph.as_default():
            with tf.device("/cpu:0"):

                const_list = []
                for ix, const in enumerate(const_iter):
                    const_name = "%s_%d" % (const_prefix, start_index + ix)
                    if verbose:
                        print("Set constant %s to %f" % (const_name, const))
                    const_list.append(tf.constant(const, dtype=float_type, name=const_name))

        return const_list

    def gen_tf_var_list(self, var_iter, var_prefix, start_index=0, verbose=False):
        """take a list or iterator of values, generate and return tensorflow Variable ops for each"""
        print ('%s Create variables %s' % (strftime("%H:%M:%S"), var_prefix))
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                var_list = []
                for ix, var in enumerate(var_iter):
                    var_name = "%s_%d" % (var_prefix, start_index + ix)
                    if verbose:
                        print("Create variable %s to %f" % (var_name, var))
                    var_op = tf.Variable(var, dtype=float_type, name=var_name)
                    self.sess.run(var_op.initializer)
                    var_list.append(var_op)

        return var_list

    def get_op_from_list(self, op_list, op_index):
        """take a list of ops, return value of op specified by op_index"""
        op = op_list[op_index]
        retval = self.sess.run([op])
        return retval

    def gen_zero_min_list(self, op_iter, op_prefix, start_index=0, verbose=False):
        """take a list or iterator of ops, generate and return an op which is max(-op, 0)
        for soft constraints > 0"""
        print ('%s Create ops for soft constraint %s > 0' % (strftime("%H:%M:%S"), op_prefix))
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                op_list = []
                for ix, op in enumerate(op_iter):
                    op_name = "%s_%d" % (op_prefix, start_index + ix)
                    if verbose:
                        print("Zero_min op %s" % op_name)
                    new_op = tf.maximum(self.zero, tf.neg(op, name="neg_%s" % op_name), name=op_name)
                    op_list.append(new_op)

        return op_list

    def gen_one_max_list(self, op_iter, op_prefix, start_index=0, verbose=False):
        """take a list or iterator of ops, generate and return an op with is max(op-1, 0)
        for soft constraints > 0"""
        print ('%s Create ops for soft constraint %s < 1' % (strftime("%H:%M:%S"), op_prefix))
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                op_list = []
                for ix, op in enumerate(op_iter):
                    op_name = "%s_%d" % (op_prefix, start_index + ix)
                    if verbose:
                        print('One_max op %s' % op_name)
                    new_op = tf.maximum(self.zero, tf.sub(op, self.one, name="one_minus_%s" % op_name),
                        name=op_name)
                    op_list.append(new_op)

        return op_list

    def gen_diff_list(self, op_iter, op_prefix, start_index=0, verbose=False):
        """generate and return an op for declining stock alloc constraint over time, max of 0 and decrease"""
        print ('%s Create ops for soft constraint, declining stock alloc %s' % (strftime("%H:%M:%S"),
                                                                                op_prefix))
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                op_list = []
                for ix, op in enumerate(op_iter):
                    if ix == 0:
                        continue
                    op_name = "%s_%d" % (op_prefix, start_index + ix)
                    if verbose:
                        print("diff op %s" % op_name)

                    new_op = tf.maximum(self.zero, tf.sub(op_iter[ix], op_iter[ix-1]))
                    op_list.append(new_op)

        return op_list

    def gen_ce(self, input_tensor, prefix, survival_tensor=None, verbose=False):
      with tf.device("/cpu:0"):
        with self.graph.as_default():
            input_length = np.float64(input_tensor.get_shape().as_list()[0])

            if verbose:
                print("%s Create ce op with gamma: %f" % (strftime("%H:%M:%S"), self.gamma))
            if self.gamma == 1.0:
                u = tf.reduce_mean(tf.log(input_tensor), name="%s_u" % prefix)
                #print(self.sess.run(u))
                if survival_tensor is not None:
                    u0 = u
                    u = tf.reduce_mean(tf.mul(u0, survival_tensor, name="%s_u_surv" % prefix),
                                       name="%s_u" % prefix)
                ce = tf.exp(u, name="%s_ce" % prefix)
                if verbose:
                    print ('%s Create CE op %f' % (strftime("%H:%M:%S"), self.sess.run(ce)))
            else:
                # for high gamma numerical error is significant, calculation is most accurate near 1
                # so divide by mean
                input_mean = tf.reduce_mean(input_tensor, name="%s_mean" % prefix)
                input_conditioned = tf.div(input_tensor, input_mean, name="%s_conditioned" % prefix)

                u1 = tf.pow(input_conditioned, self.one_minus_gamma, name="%s_u1" % prefix)
                u2 = tf.sub(u1, self.one, name="%s_u2" % prefix)
                u3 = tf.mul(u2, self.inv_one_minus_gamma, name="%s_u3" % prefix)
                u = tf.reduce_mean(u3, name="%s_u" % prefix)

                if survival_tensor is not None:
                    u4 = u
                    u = tf.reduce_mean(tf.mul(u4, survival_tensor, name="%s_u_surv" % prefix),
                        name="%s_u" % prefix)

                ce1 = tf.mul(self.one_minus_gamma, u, name="%s_ce1" % prefix)
                ce2 = tf.add(ce1, self.one, name="%s_ce2" % prefix)
                ce3 = tf.pow(ce2, self.inv_one_minus_gamma, name="%s_ce3" % prefix)
                ce = tf.mul(input_mean, ce3, name="%s_ce" % prefix)

                if verbose:
                    print ('%s Create CE op %f' % (strftime("%H:%M:%S"), self.sess.run(ce)))
            return ce


class Cohort:
    """Cohort represents experience of an individual
       - retiring in a given year
       - using the specified SafeWithdrawal model"""
    def __init__(self, model, cohort_start_year):
        self.model = model
        self.cohort_start_year = cohort_start_year
        self.name = "cohort_%d" % cohort_start_year
        self.gen_tf_ops()

    def gen_tf_ops(self, verbose=False):
        
        if verbose:
            print("%s Instantiating cohort %s" % (strftime("%H:%M:%S"), self.name))

        stock_returns = self.model.return_ops[0]
        bond_returns = self.model.return_ops[1]

        stock_allocs = self.model.allocation_ops[0]
        bond_allocs = self.model.allocation_ops[1]

        self.port_returns_list = []
        self.port_prespend_list = []
        self.port_end_vals_list = []
        self.spend_amts_list = []
        self.spend_amts_nonzero_list = []

        with self.model.graph.as_default():
          with tf.device("/cpu:0"):

            if verbose:
                print ("%s Generating %d years from %d" % (strftime("%H:%M:%S"),
                                                           self.model.ret_years,
                                                           self.cohort_start_year))

            start_year_ix = self.cohort_start_year - self.model.first_year
            for ix in range(self.model.ret_years):
                op_stock_return = stock_returns[start_year_ix + ix]
                op_stock_alloc = stock_allocs[ix]
                op_bond_return = bond_returns[start_year_ix + ix]
                op_bond_alloc = bond_allocs[ix]

                op_const_spend = self.model.const_spending_op
                op_var_spend = self.model.var_spending_ops[ix]

                op_total_real_return = tf.add(tf.mul(op_stock_alloc, op_stock_return, name="%s_stock_%d"
                                                     % (self.name, ix)),
                                              tf.mul(op_bond_alloc, op_bond_return, name="%s_bond_%d"
                                                     % (self.name, ix)),
                                              name="%s_total_return_%d" % (self.name, ix))
                self.port_returns_list.append(op_total_real_return)

                if ix == 0:
                    prev_val = self.model.start_val_op
                else:
                    prev_val = self.port_end_vals_list[ix-1]

                op_port_end_val_prespend = tf.add(prev_val,
                                                  tf.mul(prev_val, self.port_returns_list[ix],
                                                         name="%s_dolreturn_%d" % (self.name, ix)),
                                                  name="%s_prespend_%d" % (self.name, ix))
                self.port_prespend_list.append(op_port_end_val_prespend)

                desired_spend_amt = tf.add(tf.mul(op_var_spend, op_port_end_val_prespend,
                    name="%s_des_vspend_%d" % (self.name, ix)),
                                           op_const_spend,
                                           name="%s_desired_spend_amt_%d" % (self.name, ix))
                #spend minimum of tmp_spend_amt, port value
                spend_amt = tf.minimum(desired_spend_amt, op_port_end_val_prespend,
                                       name="%s_actual_spend_amt_%d" % (self.name, ix))
                self.spend_amts_list.append(spend_amt)

                op_port_end_val = tf.sub(op_port_end_val_prespend, spend_amt, name="%s_endval_%d" %
                                                                                   (self.name, ix))
                self.port_end_vals_list.append(op_port_end_val)

            #now that we've computed cohort paths we pack results into 1D Tensors to calc objective
            self.spend_amts = tf.pack(self.spend_amts_list, name="%s_spend_amts" % self.name)
            self.port_end_vals = tf.pack(self.port_end_vals_list, name="%s_port_end_vals" % self.name)

            self.mean_spending = tf.reduce_mean(self.spend_amts, name="%s_mean_spending" % self.name)
            self.sd_spending = tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(self.spend_amts,
                                                                    self.mean_spending), 2)),
                                       name="%s_sd_spending" % self.name)
            self.min_spending = tf.reduce_min(self.spend_amts, name="%s_min_spending" % self.name)
            self.max_spending = tf.reduce_max(self.spend_amts, name="%s_max_spending" % self.name)

            if self.model.gamma == 1.0:
                #spend a tiny amount even if spend is 0 so log is not NaN
                #doesn't really seem like best practice but...
                #0 spend years can't be in final solution
                #and don't want divide by zero errors if optimizer attempts one

                #chain new op off old op but keep a reference to old op around just in case
                self.spend_amts_maybe_zero = self.spend_amts
                self.spend_amts = tf.maximum(self.spend_amts_maybe_zero,
                                             self.model.very_small_amts,
                                             name="%s_actual_spend_nonzero" % self.name)
                self.total_spending = tf.reduce_sum(self.spend_amts, name="%s_total_spending_nonzero" %
                                                    self.name)
            else:
                self.total_spending = tf.reduce_sum(self.spend_amts, name="%s_total_spending" %
                                                    self.name)

            if self.model.survival is not None:
                self.ce = self.model.gen_ce_survival(self.spend_amts,
                                                     self.model.survival_tensor,
                                                     "%s_ce" % self.name)
            else:
                self.ce = self.model.gen_ce(self.spend_amts,
                                            "%s_ce" % self.name)

            #print (self.as_dataframe())

    def get_tf_ops(self):
        return self.model.start_val, self.port_returns_list, self.port_prespend_list,             self.spend_amts_list, self.port_end_vals_list, self.total_spending

    def as_dataframe(self):
        port_returns_ops = self.port_returns_list
        port_prespend_ops = self.port_prespend_list
        spend_amts_ops = self.spend_amts_list
        port_end_vals_ops = self.port_end_vals_list

        port_returns = self.model.sess.run(port_returns_ops)
        port_prespend = self.model.sess.run(port_prespend_ops)
        spend_amts = self.model.sess.run(spend_amts_ops)
        port_end_vals = self.model.sess.run(port_end_vals_ops)

        retlist = []
        for ix in range(self.model.ret_years):
            retlist.append([port_returns[ix],
                            port_prespend[ix],
                            spend_amts[ix],
                            port_end_vals[ix]
                        ])

        years = range(self.cohort_start_year, self.cohort_start_year+self.model.ret_years)
        return pd.DataFrame(retlist,
                            index = years,
                            columns=['portreturn', 'prespend', 'spend_amt', 'end_val'])

class CohortHistory:
    """represents a set of cohorts retiring in different years using a strategy,
    to enabling aggregating and summarizing their experiences"""
    def __init__(self, model, cohort_years = None):
        self.model = model
        if cohort_years is None:
            cohort_years = [year for year in range(self.model.first_year,
                                                   self.model.first_year + self.model.ret_cohorts)]

        print('%s Create cohort history, years %d to %d' % (strftime("%H:%M:%S"),
                                                            cohort_years[0], cohort_years[-1]))
        self.cohort_list = [Cohort(model, year) for year in cohort_years]
        self.total_spending_ops = [cohort.total_spending for cohort in self.cohort_list]

    def as_dataframe(self):
        """report on on each cohort by year, e.g. 1928"""
        total_spending_ops = [cohort.total_spending for cohort in self.model.cohort_history.cohort_list]
        mean_spending_ops = [cohort.mean_spending for cohort in self.model.cohort_history.cohort_list]
        sd_spending_ops = [cohort.sd_spending for cohort in self.model.cohort_history.cohort_list]
        min_spending_ops = [cohort.min_spending for cohort in self.model.cohort_history.cohort_list]
        max_spending_ops = [cohort.max_spending for cohort in self.model.cohort_history.cohort_list]
        ce_ops = [cohort.ce for cohort in self.model.cohort_history.cohort_list]

        retlist = []
        years = range(self.model.first_year, self.model.first_year + self.model.ret_cohorts)
        for year,             meanspend,             sdspend,             minspend,             maxspend,             totalspend,             ce in zip(years, self.model.sess.run(mean_spending_ops),
                      self.model.sess.run(sd_spending_ops),
                      self.model.sess.run(min_spending_ops),
                      self.model.sess.run(max_spending_ops),
                      self.model.sess.run(total_spending_ops),
                      self.model.sess.run(ce_ops)):

            retlist.append([meanspend, sdspend, minspend, maxspend, totalspend, ce])

        return pd.DataFrame(retlist, index = years,
                            columns=['mean_spend', 'sd_spend', 'min_spend', 'max_spend', 
                                     'total_spend', 'ce'])

    def spend_by_year(self):
        """report spending by year for each cohort (ret_years rows x num_cohorts)"""
        dataframes = [cohort.as_dataframe() for cohort in self.model.cohort_history.cohort_list]
        years = range(self.model.ret_years)
        cohorts = range(len(dataframes))

        retlist = []
        for ix in years:
            spendlist = [df.spend_amt.iloc[ix] for df in dataframes]
            retlist.append(spendlist)

        colnames = ["%d" % (cohort+self.model.first_year) for cohort in cohorts]
        return pd.DataFrame(retlist, index = years, columns=colnames)

    def returns_by_year(self):
        """report returns by year for each cohort (ret_years rows x num_cohorts)"""
        dataframes = [cohort.as_dataframe() for cohort in self.model.cohort_history.cohort_list]
        years = range(self.model.ret_years)
        cohorts = range(len(dataframes))

        retlist = []
        for ix in years:
            returnlist = [df.portreturn.iloc[ix] for df in dataframes]
            retlist.append(returnlist)

        colnames = ["%d" % (cohort+self.model.first_year) for cohort in cohorts]
        return pd.DataFrame(retlist, index = years, columns=colnames)

    def summarize_by_year(self):
        """report on outcomes by retirement year, e.g. retirement year 1, 2...30"""
        dataframes = [cohort.as_dataframe() for cohort in self.model.cohort_history.cohort_list]
        years = range(self.model.ret_years)
        retlist = []
        for ix in years:
            spendlist = np.array([df.spend_amt.iloc[ix] for df in dataframes])
            spend_mean = np.mean(spendlist)
            spend_sd = np.std(spendlist)
            spend_min = np.min(spendlist)
            spend_max = np.max(spendlist)
            retlist.append([spend_mean, spend_sd, spend_min, spend_max])

        return pd.DataFrame(retlist, index = years,
                            columns=['spend_mean', 'spend_sd', 'spend_min', 'spend_max'])


# Optimizer
# Create an op which is the sum of spending in all years
#  - negate it so it will be minimized
#  - add large penalty when a stock allocation is < 0 as a soft constraint
#  - add large penalty when a stock allocation is > 1 as a soft constraint
#  - add large penalty when const or var spencint is < 0 as a soft constraint
#  - result is an op which can be minimized by gradient descent

class CohortHistoryOptimize():
    def __init__(self, model):
        self.model = model
        self.best_objective = 0.0
        self.best_step = 0

        graph = self.model.graph

        with graph.as_default():
          with tf.device("/cpu:0"):

            print ('%s Create optimizer class' % strftime("%H:%M:%S"))
            print ('%s Run variable initializers' % strftime("%H:%M:%S"))
            self.model.sess.run(model.init_op)

            print('%s Create cost ops' % strftime("%H:%M:%S"))
            print('%s Sum %d ce ops' % (strftime("%H:%M:%S"), len(self.model.cohort_history.cohort_list)))
            ce_ops = [cohort.ce for cohort in self.model.cohort_history.cohort_list]
            ce_tensor = tf.pack(ce_ops, name="all_cohorts_ce_tensor")

            # ce over ret_cohorts years
            self.total_ce_op = self.model.gen_ce(ce_tensor, "all_cohorts_ce")

            print("%s Total CE spend, all cohorts: %f" % (strftime("%H:%M:%S"),
                                                          self.model.sess.run(self.total_ce_op)))
            # basic cost
            cost_op_1 = tf.neg(self.total_ce_op, name="basic_cost")
            print("%s Raw cost objective: %f" % (strftime("%H:%M:%S"), self.model.sess.run(cost_op_1)))

            cost_op_2 = tf.add(cost_op_1, model.cost_alloc_min_0_op, name="cost_add_alloc_min_0")
            print("%s Add soft constraint penalty if stock alloc < 0: %f" %
                  (strftime("%H:%M:%S"), self.model.sess.run(cost_op_2)))

            cost_op_3 = tf.add(cost_op_2, model.cost_alloc_max_1_op, name="cost_add_alloc_max_1")
            print("%s Add soft constraint penalty if stock alloc > 1: %f" % 
                  (strftime("%H:%M:%S"), self.model.sess.run(cost_op_3)))

            cost_op_4 = tf.add(cost_op_3, model.cost_vspend_min_0_op, name="cost_vspend_min_0")
            print("%s Add soft constraint penalty if var spending < 0: %f" % 
                  (strftime("%H:%M:%S"), self.model.sess.run(cost_op_4)))

            cost_op_5 = tf.add(cost_op_4, model.cost_cspend_min_0_op, name="cost_cspend_min_0")
            print("%s Add soft constraint if const spending < 0: %f" % 
                  (strftime("%H:%M:%S"), self.model.sess.run(cost_op_5)))

            self.cost_op = tf.add(cost_op_5, model.cost_alloc_decrease_op, name="cost_alloc_decrease")
            print("%s Add soft constraint if stock alloc increases in any year: %f" %
                  (strftime("%H:%M:%S"), self.model.sess.run(self.cost_op)))

            self.best_objective = -self.model.sess.run(self.cost_op)
            print("%s All inclusive objective to be minimized: %f" % (strftime("%H:%M:%S"),
                                                                      -self.best_objective))
            self.best_const_spend = self.model.sess.run(model.const_spending_op)
            self.best_var_spend = self.model.sess.run(model.var_spending_ops)
            self.best_stock_alloc = self.model.sess.run(model.allocation_ops[0])

    def run_step(self, report_steps=1):
        """run one step of optimizer
           calc gradients
           apply gradients * learning rate to each variable to descend gradient and improve objective
           increment global step to remember how many steps we've run
           if (hopefully) new objective is best to date, save params and objective"""

        _, step = self.model.sess.run([self.optimize_step,
                                       self.model.increment_step])
        self.steps_ago +=1
        cost = self.model.sess.run(self.cost_op)
        assert not(np.isnan(cost)), "Objective is nan"
        objective = - cost

        #print objective each step
        #print("objective %f best %f" %(objective, self.best_objective))
        if np.isnan(cost):
            sys.stdout.write('X')
            sys.stdout.flush()
        elif objective > self.best_objective:
            self.best_objective = objective
            self.best_const_spend = self.model.sess.run(model.const_spending_op)
            self.best_var_spend = self.model.sess.run(model.var_spending_ops)
            self.best_stock_alloc = self.model.sess.run(model.allocation_ops[0])
            self.best_step = step
            self.steps_ago = 0
            sys.stdout.write('!')
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        if step % report_steps == 0:
            sys.stdout.write("\n%s step %d objective %f best %f (%d steps ago)\n" %
                             (strftime("%H:%M:%S"),
                              step,
                              objective,
                              self.best_objective,
                              self.steps_ago))
            # print variables optimized and gradients for debugging
            # sys.stdout.write("\n")
            # var_vals = self.model.sess.run(self.model.all_var_ops)
            # print("%s Variables" % strftime("%H:%M:%S"))
            # print(var_vals)

            # grad_vals = self.model.sess.run([grad[0] for grad in self.grads])
            # print("%s Gradients" % strftime("%H:%M:%S"))
            # print(grad_vals)

            sys.stdout.flush()

            self.best_bond_alloc = pd.Series([1 - bsa for bsa in self.best_stock_alloc])
            pickle_list = [self.best_const_spend, self.best_var_spend, self.best_stock_alloc,
                           self.best_bond_alloc]
            pickle.dump( pickle_list, open( picklefile, "wb" ) )

            # every 10 report_steps show current best
            if step % (report_steps * 10) == 0:
                print ("\n#Objective: %f\n" % self.best_objective)
                print ("const_spend = %f" % self.best_const_spend)
                print ("var_spend_pcts = pd.Series(%s)" % str(self.best_var_spend))
                print ("stock_allocations = pd.Series(%s)\n" %str(self.best_stock_alloc))

    def optimize(self, learning_rate, steps):
        """create the op for the optimizer using specified learning_rate, run for specified steps"""
        self.learning_rate = learning_rate
        self.steps = steps
        self.steps_ago = 0 # how many steps since objective improved

        print("%s Objective: %f" % (strftime("%H:%M:%S"), self.best_objective))
        print("%s Constant spending: %f" % (strftime("%H:%M:%S"), self.best_const_spend))
        print("%s Variable spending by year" % strftime("%H:%M:%S"))
        print(self.best_var_spend)
        print("%s Stock allocation by year" % strftime("%H:%M:%S"))
        print(self.best_stock_alloc)

        with self.model.graph.as_default():
            with tf.device("/cpu:0"):

                # minimize op
                print('%s Create optimizer (learning rate %.12f)' % (strftime("%H:%M:%S"),
                                                                     self.learning_rate))
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.grads = self.optimizer.compute_gradients(self.cost_op)
                self.optimize_step = self.optimizer.apply_gradients(self.grads)
                # following line is equivalent to previous 2 lines
                # self.optimize_step = self.optimizer.minimize(self.cost_op)

        print('%s Create optimizer op and run %d steps' % (strftime("%H:%M:%S"), self.steps))
        for i in range(self.steps):
            self.run_step()


learning_rate = 0.0000001
fileprefix = "gamma_8"
picklefile = '%s.pickle' % fileprefix
csvfile = "summary_%s.csv" % fileprefix
yearsfile = "years_%s.csv" % fileprefix
returnsfile = "retyears_%s.csv" % fileprefix


max_steps_unimproved = 500
gamma = 8

print('%s Start optimization session' % strftime("%H:%M:%S"))
print('%s learning_rate: %.12f steps %d picklefile %s' % (strftime("%H:%M:%S"), learning_rate, max_steps_unimproved, picklefile))

#print("opening picklefile %s" % picklefile)
#const_spend, var_spend_pcts, stock_allocations, bond_allocations  = pickle.load( open(picklefile, "rb" ) )

print ("const spend_pct: %f" % const_spend_pct)
print ("variable spend:")
print (var_spend_pcts)
print ("stock allocation:" )
print (stock_allocations)

model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = gamma,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

print('%s Summary by cohort' % strftime("%H:%M:%S"))
print(model.cohort_history.as_dataframe())
all_years = model.cohort_history.spend_by_year()
all_years.to_csv(yearsfile, format="%.18f")
ret_years = model.cohort_history.returns_by_year()
ret_years.to_csv(returnsfile, format="%.18f")

summary = model.cohort_history.summarize_by_year()
print(summary)
summary.to_csv(csvfile, format="%.18f")

# run optimizer
# set an initial learning rate that improves objective by a reasonable amount each step

model.optimizer = CohortHistoryOptimize(model)
model.optimizer.optimize(learning_rate, steps=1)

# continue optimizing without re-initializing vars or optimizer
# reduce learning rate if no improvement for a while
# end when learning rate is too small to make significant improvement

max_steps = 1001 # don't run for hours because notebook will eventually crash
report_steps = 50
learning_rate = model.optimizer.learning_rate

for i in range(max_steps):
    model.optimizer.run_step(report_steps=report_steps)
    if model.optimizer.steps_ago >= max_steps_unimproved: # no improvement for too long
        break

const_spend = model.optimizer.best_const_spend
var_spend_pcts = pd.Series(model.optimizer.best_var_spend)
stock_allocations = pd.Series(model.optimizer.best_stock_alloc)
bond_allocations   = 1 - stock_allocations
pickle_list = [const_spend, var_spend_pcts, stock_allocations, bond_allocations]
pickle.dump( pickle_list, open( picklefile, "wb" ) )


# This is a good solution, objective ~4.7 after running many times at decreasing learning rates

const_spend_pct = 0.018033736915
var_spend_pcts = pd.Series([0.027420789200650691, 0.028952687799914788, 0.029987640366612587, 0.030563608815231274, 0.031468009954100477, 0.0327892165458714, 0.033812194391987072, 0.035195378137199251, 0.037327252304951188, 0.039532656754857148, 0.041711339480165768, 0.044205216561892961, 0.047114695564658048, 0.049375968207551808, 0.051345728157015109, 0.05421706340461499, 0.057174919670928206, 0.060803826336315064, 0.065058176584335367, 0.069765865219855477, 0.073504852805054741, 0.079201426822341936, 0.087219117566384868, 0.095745358220279422, 0.10919882479478223, 0.12781818289102989, 0.15488310518171028, 0.20138642112957703, 0.29915169759218702, 1.0])
stock_allocations = pd.Series([0.85995594522555152, 0.85919296367537157, 0.85473424640534235, 0.85205353144872398, 0.84754303090697891, 0.84731613048078414, 0.84674123511449917, 0.83207329884146453, 0.81886882987838872, 0.81886882879077094, 0.81787428397721995, 0.81082294011238876, 0.80935846467721007, 0.80915247713021987, 0.80869082405588244, 0.79235039525216477, 0.78697518924653442, 0.78566021347854842, 0.77606352982164872, 0.77226108408967586, 0.76418701504255893, 0.75957408240400748, 0.7506635662237392, 0.74210994715180123, 0.68810212447786767, 0.68212008875931995, 0.66794616058167322, 0.66270963157142881, 0.65979617905489518, 0.61499306500466799])

model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 8.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

print('%s Summary by cohort' % strftime("%H:%M:%S"))
print(model.cohort_history.as_dataframe())


df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.show()


df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(df_summary['const_spend'], label="constant spending")
ax.plot(df_summary['var_spend'], label="variable spending")
ax.plot(df_summary['stocks'], label="stocks")
ax.plot(df_summary['bonds'], label="bonds")

plt.ylabel("% of portfolio (starting portfolio for const spending)")
plt.xlabel("Retirement year")

#legend causes probs for plotly
#leg = ax.legend(loc="upper left", fancybox=True)
#leg.get_frame().set_alpha(0.3)

plt.show()

plot_url = py.iplot_mpl(fig, filename='chart')
plot_url

from decimal import *
getcontext().prec = 32

def u (cashflow, gamma):
    d_cashflow = Decimal(cashflow)
    d_gamma = Decimal(gamma)
    d_one = Decimal('1.00000000000000000000000000000000')
    
    if gamma != 1:
        one_minus_gamma = d_one - d_gamma
        u = (d_cashflow ** one_minus_gamma - d_one) / one_minus_gamma
        return u
    else:
        return d_cashflow.ln()


x = np.linspace(0,10,1001)
y0 = np.array([u(z, 0.0) for z in x])
y1 = np.array([u(z, 1.0) for z in x])
y2 = np.array([u(z, 2.0) for z in x])
y4 = np.array([u(z, 4.0) for z in x])
y8 = np.array([u(z, 8.0) for z in x])
y16 = np.array([u(z, 16.0) for z in x])
y32 = np.array([u(z, 32.0) for z in x])

# note, at high gamma, can barely distinguish 9.9 from 10
# even with extra decimal places vs. float64
y32

plt.axis([ -0.2, 6.2, -3.0, 3,])

plt.plot(x, y0, color='k', label = r'$\gamma = 0$')
plt.plot(x, y1, color='b', label = r'$\gamma = 1$')
plt.plot(x, y2, color='r', label = r'$\gamma = 2$')
plt.plot(x, y4, color='g', label = r'$\gamma = 4$')
plt.plot(x, y8, color='purple', label = r'$\gamma = 8$')
plt.plot(x, y16, color='cyan', label = r'$\gamma = 16$')
plt.plot(x, y32, color='orange', label = r'$\gamma = 32$')

plt.ylabel("CRRA utility")
plt.xlabel("Cash flow ")
plt.legend(loc="lower right", ncol=2, bbox_to_anchor=[1, 0])


# gamma = 0

const_spend_pct = 0.000000
var_spend_pcts = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
stock_allocations = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 0.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

# gamma = 1
#Objective: 9.308091

const_spend_pct = 0.000210991438
var_spend_pcts = pd.Series([0.033091538348404496, 0.034238168285506382, 0.035469702440122583, 0.036794378152860914, 0.038221756778048857, 0.03976288047503998, 0.041430947662087859, 0.043242686669740876, 0.045216839687742649, 0.047378561748440171, 0.04975812222825144, 0.052387429460561782, 0.055309651974463532, 0.058579229736857016, 0.062260304236144169, 0.066431027576827867, 0.071195216736901532, 0.076691471776952069, 0.083104783014253336, 0.090686442187799754, 0.099783243431139951, 0.11089638349654848, 0.12478101422016996, 0.14262272988447125, 0.16639735490582469, 0.19966642400185314, 0.24955549906491153, 0.33270239606716523, 0.49854063104140045, 1.0])
stock_allocations = pd.Series([0.99999999932534844, 0.99997650919314807, 0.99995061726681489, 0.99992473029357398, 0.99933647980151297, 0.99915824051192981, 0.99801222176142057, 0.9979130315525, 0.99767125200215423, 0.99762089448679248, 0.99748512848933513, 0.99744284142382889, 0.99734447386668101, 0.99709686697171851, 0.99700226658406843, 0.99670993806194219, 0.99638045411265985, 0.99628135924967531, 0.99610184313223671, 0.99588814803982129, 0.99574876269478818, 0.99557475808121887, 0.99508658081662371, 0.99501534744474973, 0.99468563684794653, 0.98122175151927205, 0.97403705686938635, 0.9563148825731429, 0.93175496886826248, 0.91435187362375314])


model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 1.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

# gamma = 2

#Objective: 7.232013

const_spend_pct = 0.006633196051
var_spend_pcts = pd.Series([0.041651782115649724, 0.04298351383672569, 0.044307405688720158, 0.045644563126512307, 0.046935371105069679, 0.048256594099694364, 0.049876320240762828, 0.051675640492043345, 0.053725939553432397, 0.056085607010629972, 0.058531950342040277, 0.061306089393194735, 0.064303573280149864, 0.067511781568926968, 0.070864795122752594, 0.074644853152054047, 0.079354357195997025, 0.084874368957208082, 0.091405892559806134, 0.098722229027103148, 0.10732707056538419, 0.11779806370242611, 0.13130833549322588, 0.1483069190268195, 0.17052212615693424, 0.20140944216392345, 0.24724327540518604, 0.32492166343973394, 0.48312975250382012, 1.0])
stock_allocations = pd.Series([0.99128140981577162, 0.9890186164406034, 0.9889931746406847, 0.98899317444042156, 0.98897294365892874, 0.98892702392459531, 0.98017623383520169, 0.97434793556657606, 0.96296843154230494, 0.96175969914252735, 0.96164944382766815, 0.95833816100296942, 0.95788049761186367, 0.95748409443075122, 0.94935349502063981, 0.94491171918437289, 0.93545099514061691, 0.9305036848920506, 0.92943528021152122, 0.92302678056280996, 0.92027686390605101, 0.91845762356824345, 0.91278569545635735, 0.91231424523945759, 0.90537710239072444, 0.89686206541538249, 0.88878472270546383, 0.87921904341544299, 0.86565496875257597, 0.84597976461898239])



model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 2.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.axis([ 0, 30, 0, 60,])

plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

# gamma = 4
#Objective: 5.653304

const_spend_pct = 0.013135043394
var_spend_pcts = pd.Series([0.035822978028495614, 0.037142880037028099, 0.038232214787301988, 0.039230415975672818, 0.040201921370750057, 0.04128306632974512, 0.042562738008764228, 0.044203187483426658, 0.04612997210464486, 0.048403252082914246, 0.050913960441316269, 0.053804872403093118, 0.056961354837341847, 0.059937273709488398, 0.062775232619198637, 0.066025078674551443, 0.070108404481856915, 0.075376643015682129, 0.081397919560765669, 0.087812983760457097, 0.094573649661557693, 0.10292659618285707, 0.11353980019663899, 0.12569730816023159, 0.14198879320676885, 0.16414995001568367, 0.19648563495632851, 0.25250659931593217, 0.3706052129208936, 1.0])
stock_allocations = pd.Series([0.90099032199651863, 0.89924731857030904, 0.89924731828348214, 0.89876830969341182, 0.89770857617754174, 0.89503554351652825, 0.89232180232291247, 0.88505129512940295, 0.8648266951006971, 0.86372639925213479, 0.85995907966929552, 0.85302430311396027, 0.85289466074670783, 0.85270233833392206, 0.85157005570710653, 0.84601302356094343, 0.83930541580936513, 0.83354308642897557, 0.82595372913293841, 0.82174793960063786, 0.80903946044432207, 0.80583306771707164, 0.79764354604204957, 0.79580225813480721, 0.75787634532529125, 0.75092813511924261, 0.73890315821297692, 0.73198221026156862, 0.72778224605194297, 0.69545344502127571])


model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 4.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.axis([ 0, 30, 0, 60,])

plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

# gamma = 6

#Objective: 5.056278

const_spend_pct = 0.015637933571
var_spend_pcts = pd.Series([0.031606123286676821, 0.033096808496424615, 0.034176874786077449, 0.034949059621799367, 0.035890789552150887, 0.037116351576046536, 0.038256812396706585, 0.039784074922038948, 0.041850514856008901, 0.044113104719702513, 0.046479531291527841, 0.049188553520785189, 0.052237088536900282, 0.054826487679491116, 0.05719659838516862, 0.060236417433033075, 0.06372908174792162, 0.068170731957061817, 0.073291375378079365, 0.078832696364871821, 0.084046153747339689, 0.091046547868110006, 0.10034482939483094, 0.11066289255761518, 0.1255268715113966, 0.14590561544900946, 0.17559280570819452, 0.22684587500024678, 0.33478357325498115, 1.0])
stock_allocations = pd.Series([0.88033446277449279, 0.87918369248956052, 0.87699906807696382, 0.87541489848777843, 0.87254543385561067, 0.87112127695141239, 0.8696038278716699, 0.85852400031522713, 0.84145611984667457, 0.84145611982772339, 0.83898260822198845, 0.83191026515233402, 0.83112108486165093, 0.83100258791320347, 0.83025217396697193, 0.81919450523261073, 0.81311531977557161, 0.809624259366025, 0.80101985702945411, 0.79700856489108074, 0.78660821862681019, 0.7827212663418508, 0.77416817507231639, 0.76897655138636212, 0.72297797819724985, 0.7165312149934886, 0.70342618929327727, 0.69734141791783077, 0.69379260335071868, 0.65522743833716124])

model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 6.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.axis([ 0, 30, 0, 60,])

plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

#gamma = 8

#Objective: 4.344292

const_spend_pct = 0.018033736915
var_spend_pcts = pd.Series([0.027420789200650691, 0.028952687799914788, 0.029987640366612587, 0.030563608815231274, 0.031468009954100477, 0.0327892165458714, 0.033812194391987072, 0.035195378137199251, 0.037327252304951188, 0.039532656754857148, 0.041711339480165768, 0.044205216561892961, 0.047114695564658048, 0.049375968207551808, 0.051345728157015109, 0.05421706340461499, 0.057174919670928206, 0.060803826336315064, 0.065058176584335367, 0.069765865219855477, 0.073504852805054741, 0.079201426822341936, 0.087219117566384868, 0.095745358220279422, 0.10919882479478223, 0.12781818289102989, 0.15488310518171028, 0.20138642112957703, 0.29915169759218702, 1.0])
stock_allocations = pd.Series([0.85995594522555152, 0.85919296367537157, 0.85473424640534235, 0.85205353144872398, 0.84754303090697891, 0.84731613048078414, 0.84674123511449917, 0.83207329884146453, 0.81886882987838872, 0.81886882879077094, 0.81787428397721995, 0.81082294011238876, 0.80935846467721007, 0.80915247713021987, 0.80869082405588244, 0.79235039525216477, 0.78697518924653442, 0.78566021347854842, 0.77606352982164872, 0.77226108408967586, 0.76418701504255893, 0.75957408240400748, 0.7506635662237392, 0.74210994715180123, 0.68810212447786767, 0.68212008875931995, 0.66794616058167322, 0.66270963157142881, 0.65979617905489518, 0.61499306500466799])



model = SafeWithdrawalModel(returns_list = [real_stocks, real_bonds],
                            names_list = ["stocks","bonds"],
                            allocations_list = [stock_allocations, bond_allocations],
                            start_val = 100.0,
                            const_spend_pct = const_spend_pct,
                            var_spend_pcts = var_spend_pcts,
                            gamma = 8.0,
                            survival=None
)

# generate cohorts
model.cohort_history = CohortHistory(model)

df_summary = model.cohort_history.summarize_by_year()
df_years = model.cohort_history.spend_by_year()

from pylab import rcParams
rcParams['figure.figsize'] = 8, 6

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    plt.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label='_nolegend_')


plt.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
plt.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
plt.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
plt.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=1, label='+1 SD')
plt.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=1, label='-1 SD')
plt.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
plt.legend(loc="upper left", bbox_to_anchor=[0, 1])
plt.axis([ 0, 30, 0, 60,])

plt.show()

df_summary['const_spend'] = const_spend_pct * 100
df_summary['var_spend'] = var_spend_pcts * 100
df_summary['stocks'] = stock_allocations * 100
df_summary['bonds'] = 100 - df_summary['stocks']
cols = ['const_spend', 'var_spend', 'stocks', 'bonds', 'spend_mean', 'spend_min', 'spend_max']
df_summary[cols]

# make a plotly chart

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

from matplotlib import colors
mycolors =  list(colors.cnames)
random.shuffle(mycolors)

for ix in range(model.first_year, model.first_year + model.ret_cohorts):
    ax.plot(df_years[str(ix)], linewidth=1, color=mycolors[ix-model.first_year], alpha=0.5, label=str(ix))


ax.plot(df_summary.spend_mean, color='k', linewidth=2, label='Mean')
ax.plot(df_summary.spend_max, color='g', linewidth=2, label='Best')
ax.plot(df_summary.spend_min, color='r', linewidth=2, label='Worst')
ax.plot(df_summary.spend_mean + df_summary.spend_sd, color='b', linewidth=2, label='+1 SD')
ax.plot(df_summary.spend_mean - df_summary.spend_sd, color='b', linewidth=2, label='-1 SD')
ax.fill_between(df_summary.index, df_summary.spend_mean + df_summary.spend_sd, 
                 df_summary.spend_mean - df_summary.spend_sd, color='blue', alpha='0.1')

plt.ylabel("Annual spending as % of starting portfolio")
plt.xlabel("Retirement year")
#plotly pukes on legend
#ax.legend(loc="upper left", bbox_to_anchor=[0, 1])
ax.axis([ 0, 30, 0, 60,])

plt.show()

plot_url = py.iplot_mpl(fig, filename='chart')
plot_url



