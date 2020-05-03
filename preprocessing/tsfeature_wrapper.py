import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import rpy2
import torch
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2 import robjects


def init_tsfeatures():
    """
    Install tsfeatrues R package https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html
    :return: the variable corresponding to the package
    """
    utils = importr('utils')
    utils.install_packages('tsfeatures')
    return importr('tsfeatures')

# Need import for next function
tsfeatures = init_tsfeatures()


def calc_ts_feature(data, name):
    """
    Implementation of wrapper around the R package tsfeatures see :
    https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html for more details
    :param data:
    :param name:
    :return:
    """
    if name == 'entropy':
        return data.parallel_apply(lambda x: tsfeatures.entropy(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'lumpiness':
        return data.parallel_apply(lambda x: tsfeatures.lumpiness(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'stability':
        return data.parallel_apply(lambda x: tsfeatures.stability(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'max_level_shift':
        return data.parallel_apply(lambda x: tsfeatures.max_level_shift(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'max_var_shift':
        return data.parallel_apply(lambda x: tsfeatures.max_var_shift(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'max_kl_shift':
        return data.parallel_apply(lambda x: tsfeatures.max_kl_shift(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'crossing_points':
        return data.parallel_apply(lambda x: tsfeatures.crossing_points(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'flat_spots':
        return data.parallel_apply(lambda x: tsfeatures.flat_spots(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'hurst':
        return data.parallel_apply(lambda x: tsfeatures.hurst(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'stl_features':
        def extract_stl_features(x):
            stl_features = tsfeatures.stl_features(robjects.FloatVector(x.values))
            return [stl_features[i] for i in range(len(stl_features))]
        stl_features = data.parallel_apply(lambda x: extract_stl_features(x), axis=1)
        return pd.DataFrame(stl_features.explode().values.reshape((data.shape[0], -1)))
    if name == 'unitroot_kpss':
        return data.parallel_apply(lambda x: tsfeatures.unitroot_kpss(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'nonlinearity':
        return data.parallel_apply(lambda x: tsfeatures.nonlinearity(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'diff1_acf1':
        return data.parallel_apply(lambda x: tsfeatures.acf_features(robjects.FloatVector(x.values))[2], axis=1)
    if name == 'x_acf1':
        return data.parallel_apply(lambda x: tsfeatures.acf_features(robjects.FloatVector(x.values))[0], axis=1)
    if name == 'x_acf10':
        return data.parallel_apply(lambda x: tsfeatures.acf_features(robjects.FloatVector(x.values))[1], axis=1)
    if name == 'diff1_acf1':
        return data.parallel_apply(lambda x: tsfeatures.acf_features(robjects.FloatVector(x.values))[2], axis=1)
    if name == 'diff1_acf10':
        return data.parallel_apply(lambda x: tsfeatures.acf_features(robjects.FloatVector(x.values))[3], axis=1)