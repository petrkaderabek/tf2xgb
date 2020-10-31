import numpy as np
import xgboost as xgb
import tensorflow as tf
import pandas as pd

class FlatVsNDimArrayConverter:
    """XGBoost uses 1D arrays for targets and predictions. On the other hand, 
    used-defined TensorFlow pooling of predictions will be done on multiple
    dimensions. 
    
    Typically, target will be on group level with 1D shape (#groups), while
    input features of XGBoost on the level of individual observations
    with 2D shape (#obs, #feats). This will result in XGBoost predictions
    with 1D shape (#obs). This class will convert them to 2D with shape
    (#groups, max #obs per group). This is a convenient format for user-defined
    TensorFlow pooling and loss function: Pooling over the second dimension 
    will turn the shape of predictions to 1D (#groups), which is already 
    equal to the shape of the target. Predictions and target in the same
    shape can enter a TensorFlow loss function, which provides a scalar value.
    """

    def __init__(self, ragged_nested_index_lists):
        """Initialize with ragged nested lists of indices to flat array, 
        which will be used to convert between the flat and multidimensional 
        arrays
        """
        self.ragged_nested_index_lists = ragged_nested_index_lists
        self.nd_shape = self._get_shape_of_ragged_nested_lists(self.ragged_nested_index_lists)

    def nd2flat(self, nd_array, flat_shape, fill_value=0.0):
        """Flatten the multidimensional array nd_array to the shape flat_shape.
        Values of the output array, which are not filled because their indices 
        are not in self.ragged_nested_index_lists, will be filled with fill_value.
        """
        flat = np.full(flat_shape, fill_value)
        self._flatten_nd_array_by_nested_indices(flat, self.ragged_nested_index_lists, nd_array)
        return flat
    
    def flat2nd(self, flat, nd_shape=None, fill_value=0.0):
        """Convert the flat array `flat` to the shape nd_shape. If nd_shape
        is None (default), the shape inferred from self.ragged_nested_index_lists
        is used.

        Values of the output array, which are not filled because of the
        ragged self.ragged_nested_index_lists, will be filled with fill_value.
        """
        if nd_shape is None:
            nd_shape = self.nd_shape
        nd_array = np.full(nd_shape, fill_value)
        self._set_nd_array_by_nested_indices(nd_array, self.ragged_nested_index_lists, flat)
        return nd_array

    def _get_shape_of_ragged_nested_lists(self, l, _level=0, _shape=[]):
        """Infers shape of a nested list or multidimensional array
        needed to fit the ragged nested list l.

        Examples:
        get_shape_of_ragged_nested_lists([1, 2, 3])==[3]
        get_shape_of_ragged_nested_lists([[1], [2], [3, 33]])==[3, 2]
        get_shape_of_ragged_nested_lists([[1], [2], [3, [33]]])==[3, 2, 1]
        """
        if _level == 0:
            # create new instance of list, do not inherit any existing list from 
            # previous runs
            _shape = []
        if isinstance(l, list):
            if len(_shape)<=_level:
                _shape += [0]
            _shape[_level] = max(_shape[_level],len(l))
            for ll in l:
                self._get_shape_of_ragged_nested_lists(ll, _level=_level+1, _shape=_shape)
        return _shape

    def _set_nd_array_by_nested_indices(self, output_nd_array, nested_list_of_indices, input_array):
        """
        Parameters:
        = output_nd_array: the array the results will be stored in; this
        array has to be created before calling this function and be pre-set
        with the default values, which will be kept wherever no data
        from input_array will be filled.
        = nested_list_of_indices: the nested list of indices to input_array;
        in the example of 2D nested_list_of_indices, consider 
        nested_list_of_indices[i, j] = k; this means that output_nd_array[i,j][:]
        will be set to vector (or array) input_array[k]
        = input_array: the array, where the first coordinate corresponds to the
        values in nested_list_of_indices and the remaining dimension(s) 
        correspond(s) to the last dimensions of output_nd_array

        Example:
        output_nd_array = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        nested_list_of_indices = [2, 1, 0]
        input_array = [[1, 1], [2, 2], [3, 3]]
        Result (stored in output_nd_array): [[3, 3], [2, 2], [1, 1], [0, 0], [0, 0]]
        """
        shape_of_copied_element = input_array.shape[1:]
        element_is_scalar = len(shape_of_copied_element)==0
        out_shape_of_copied_element = output_nd_array.shape[-len(shape_of_copied_element):]
        assert element_is_scalar or \
            shape_of_copied_element == out_shape_of_copied_element, \
            "The last dimensions of input_array and output_nd_array" \
            f"must match but found {shape_of_copied_element} " \
            f"and {out_shape_of_copied_element}."
        for i,ll in enumerate(nested_list_of_indices):
            if isinstance(ll, int):
                output_nd_array[i] = input_array[ll]
            elif isinstance(ll, list):
                self._set_nd_array_by_nested_indices(output_nd_array[i], ll, input_array)
            else:
                raise ValueError(f"Expected value of type int or list in list of indices "
                                f"nested_list_of_indices but {nested_list_of_indices} "
                                f"found.")
                
    def _flatten_nd_array_by_nested_indices(self, output_array, nested_list_of_indices, input_nd_array):
        """
        Parameters:
        = output_array: the array the results will be stored in; the first 
        coordinate corresponds to the values in nested_list_of_indices 
        and the remaining dimension(s) correspond(s) to the last dimensions of 
        input_nd_array; this array has to be created before calling this function 
        and be pre-set with the default values, which will be kept wherever no 
        data from input_array will be filled.
        = nested_list_of_indices: the nested list of indices to output_array;
        in the example of 2D nested_list_of_indices, consider 
        nested_list_of_indices[i, j] = k; this means that output_array[k]
        will be set to vector (or array) input_nd_array[i,j][:]
        = input_nd_array: the input array with nested structure


        Example:
        output_array = [[0, 0], [0, 0], [0, 0]]
        nested_list_of_indices = [2, 1, 0]
        input_nd_array = [[3, 3], [2, 2], [1, 1], [0, 0], [0, 0]]
        Result (stored in output_array): [[1, 1], [2, 2], [3, 3]]
        """
        shape_of_copied_element = output_array.shape[1:]
        element_is_scalar = len(shape_of_copied_element)==0
        in_shape_of_copied_element = input_nd_array.shape[-len(shape_of_copied_element):]
        assert element_is_scalar or \
            shape_of_copied_element == in_shape_of_copied_element, \
            "The last dimensions of output_array and input_nd_array " \
            f"must match but found {shape_of_copied_element} " \
            f"and {in_shape_of_copied_element}."
        for i,ll in enumerate(nested_list_of_indices):
            if isinstance(ll, int):
                output_array[ll] = input_nd_array[i]
            elif isinstance(ll, list):
                self._flatten_nd_array_by_nested_indices(output_array, ll, input_nd_array[i])
            else:
                raise ValueError(f"Expected value of type int in list of indices "
                                f"nested_list_of_indices but {nested_list_of_indices} "
                                f"found.")
                                
                        
def get_ragged_nested_index_lists(df, id_col_list):
    """
    Gets the ragged nested lists of indices (= row numbers of `df`).
    Hierarchy in the nesting is set up by the `df` columns with names
    in `id_col_list`.

    Inputs:
    = df: Pandas Data Frame with the sample. It ias to contain columns 
    listed in `id_col_list`.
    = id_col_list: list of columns to `df` which correspond to the levels
    of nesting in the resulting index list. Higher-level groups have to 
    be mentioned first, e.g. ['grp_id', 'subgrp_id'].

    Returns:
    Pandas DF with two columns:
    = copy of df[id_col_list[0]]
    = column `_row_` containing nested list of row numbers, which is input to 
    decorator `xgb_tf_loss()`.

    Example:
    id_col_list = ['grp_id', 'subgrp_id']
    The result has 3 levels of nesting: (#grp_ids, subgrp_id within grp_ids,
    individual observation within subgrp_id)
    """
    df_idx = df[id_col_list].copy()
    df_idx['_row_'] = list(range(len(df_idx)))
    remaining_cols = list(id_col_list)
    while remaining_cols:
        c = remaining_cols.pop()
        df_idx = df_idx.groupby(c).agg({'_row_':lambda x: list(x), **{rc:max for rc in remaining_cols}}).reset_index()
    return df_idx
    
    
def gen_random_dataset(n, n_subgrp, n_grp, beta, sigma):
    """
    Generate random Pandas Data Frame with `n` observations split to `n_subgrp`
    distinct subgroups described by column 'subgrp_id' and `n_grp` distinct 
    groups described by column 'grp_id'. The target in column 'y' is
    linear combination of feature vector in column 'X' with true coefficient 
    vector `beta` and standard error `sigma`. Intercept is zero.
    """
    assert n >= n_subgrp
    assert n_subgrp >= n_grp

    def gen_rand_sample(n_obs, beta, sigma):
        n_feats = len(beta)
        X = np.random.randn(n_obs, n_feats)
        epsilon = np.random.randn(n_obs)*sigma
        y = np.matmul(X, beta) + epsilon
        return X,y

    def gen_rand_grp_id(n_obs, n_grp, prefix='GRP'):
        grp_num = np.concatenate((
            np.arange(n_grp, dtype='I'),
            np.random.randint(0,n_grp,size=n_obs-n_grp)
            ))
        grp = np.asarray([f'{prefix}{i}' for i in grp_num])
        return grp

    X,y = gen_rand_sample(n, beta, sigma)
    df = pd.DataFrame(data={f'_row_':list(range(len(y))), 'X':X.tolist(), 'y':y})
    if n_subgrp:
        subgrp_ids = gen_rand_grp_id(n, n_subgrp, prefix='SUBGRP')
        subgrp_ids_distinct = list(set(subgrp_ids.tolist()))
        df['subgrp_id'] = subgrp_ids
        if n_grp:
            grp_ids = gen_rand_grp_id(len(subgrp_ids_distinct), n_grp, prefix='GRP')
            subgrp2grp_id_dict = {k:v for k,v in zip(subgrp_ids_distinct, grp_ids)}
            df['grp_id'] = df['subgrp_id'].map(subgrp2grp_id_dict)
    return df
    
    
@tf.function
def tf_d_loss_single_input(tf_loss_fn, target, preds_nd, mask):
    """Calculate first and second derivatives of tf_loss_fn with respect to
    preds_nd with parameters target. The differentiation is done using 
    TensorFlow.

    Inputs:
    tf_loss_fn: TensorFlow function with two numpy array inputs: 1D target
    with shape (#groups) and multidimensional predictions with the 
    first dimension of length #groups. This function internally performs pooling 
    of predictions to the 1D shape (#groups), compares them with target and
    returns a scalar loss value. This output reflects MEAN of losses,
    which is the output of e.g. tf.keras.losses.mean_squared_error().
    The MEAN is translated to SUM later in this function because of the 
    compatibility with XGB custom objective function.
    """
    t = target
    x = preds_nd
    msk = mask
    shp = tf.shape(x)
    shp_dx = tf.concat([shp[:1],tf.ones([tf.rank(x)-1], dtype=shp.dtype)],
                         axis=0)
    delta_x = tf.zeros(shp_dx, dtype=x.dtype)
    with tf.GradientTape() as g:
        g.watch(delta_x)
        with tf.GradientTape() as gg:
            gg.watch(delta_x)
            loss = tf_loss_fn(t, x+delta_x*msk)
            # to be sure the resulting loss is a scalar, calculate MEAN
            # of possible several losses first to obtain MEAN loss.
            # for compatibility with XGB, make SUM from the MEAN by 
            # multiplying with number of observations (elements in target)
            scale = tf.size(t, out_type=loss.dtype)
            loss = tf.math.reduce_mean(loss)*scale
        dloss_dx = gg.gradient(loss, delta_x)
    d2loss_dx2 = g.batch_jacobian(dloss_dx, delta_x)
    d2loss_dx2 = tf.reshape(d2loss_dx2, tf.shape(dloss_dx))
    return dloss_dx, d2loss_dx2


def tf_d_loss(tf_loss_fn, target, preds_nd):
    """
    Returns 1st and 2nd order derivatives of the scalar loss resulting from
    tf_loss_fn() with respect to preds_nd.

    Inputs:
    tf_loss_fn: TensorFlow function with two numpy array inputs: 1D target
    with shape (#groups) and multidimensional predictions with the 
    first dimension of length #groups. This function internally performs pooling 
    of predictions to the 1D shape (#groups), compares them with target and
    returns a scalar loss value. This output reflects MEAN of losses,
    which is the output of e.g. tf.keras.losses.mean_squared_error().
    The MEAN is translated to SUM later in this function because of the 
    compatibility with XGB custom objective function.
    = target: tensor with group-level targets
    = preds_nd: n-dimensional tensor of (reshaped) XGBoost predictions,
    which is fed into tf_loss_fn() for it to perform pooling and loss 
    calculation.

    Result:
    tuple of 1st derivatives tensor and 2nd derivatives tensor. Both these 
    tensors have the same shape as the input preds_nd tensor.
    """
    # iterate over possible masks keeping always only one element of preds_nd[i]
    # this is to calculate partial derivatives of loss w.r.t all preds_nd[i]
    # elements; unfortunately no solution, how to calculate it without this 
    # loop, was found
    mask_shape = preds_nd.shape[1:]
    n_masks = np.prod(mask_shape)
    grad_list = []
    hess_list = []
    for i in range(n_masks):
        mask = np.zeros([n_masks], dtype=preds_nd.dtype)
        mask[i] = 1
        mask = mask.reshape(mask_shape)
        g, h = tf_d_loss_single_input(tf_loss_fn, 
                                      tf.constant(target, dtype=tf.float64), 
                                      tf.constant(preds_nd, dtype=tf.float64), 
                                      tf.constant(mask, dtype=tf.float64))
        grad_list.append(g.numpy())
        hess_list.append(h.numpy())
    grad = np.stack(grad_list, axis=-1).reshape(preds_nd.shape)
    hess = np.stack(hess_list, axis=-1).reshape(preds_nd.shape)
    return grad, hess


def xgb_tf_loss(ragged_nested_index_lists, target):
    """Decorator of custom TensorFlow pooling&loss function tf_loss_fn.
    It produces the custom objective function, which is input to XGBoost.

    Decorated function:
    = tf_loss_fn: TensorFlow function with two numpy array inputs: 1D target
    with shape (#groups) and multidimensional predictions with the 
    first dimension of length #groups. This function internally performs pooling 
    of predictions to the 1D shape (#groups), compares them with target and
    returns a scalar loss value. This output reflects MEAN of losses,
    which is the output of e.g. tf.keras.losses.mean_squared_error().
    The MEAN is translated to SUM later in this function because of the 
    compatibility with XGB custom objective function.

    IMPORTANT:
    Missing values in the multidimensional predictions are denoted by np.nan and 
    have to be taken care of by tf_loss_fn function body. They occur simply 
    because the multidimensional predictions tensor has typically much more 
    elements that the original flat predictions vector from XGBoost.

    Inputs:
    = ragged_nested_index_lists: the nested list of indices to XGB predictions
      vector; in the example of 2D ragged_nested_index_lists, consider 
      ragged_nested_index_lists[i, j] = k; this means that xgb_predictions[k]
        = predictions_input_to_tf_pooling_and_loss[i, j] 
    = target: tensor with group-level targets
    """
    array_converter = FlatVsNDimArrayConverter(ragged_nested_index_lists)
    def decorator(tf_loss_fn):
        def xgb_obj_fn(preds, dtrain, **kwargs):
            preds_nd = array_converter.flat2nd(preds, fill_value=np.nan)
            grad_nd, hess_nd = tf_d_loss(tf_loss_fn, target, preds_nd)
            grad = array_converter.nd2flat(grad_nd, preds.shape, fill_value=0.0)
            hess = array_converter.nd2flat(hess_nd, preds.shape, fill_value=0.0)
            return grad, hess
        return xgb_obj_fn
    return decorator
    
    

