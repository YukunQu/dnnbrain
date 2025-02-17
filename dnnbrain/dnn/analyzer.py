import numpy as np

from dnnbrain.dnn import io as dio
from dnnbrain.dnn.models import dnn_truncate
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from scipy.signal import convolve, periodogram
from sklearn.decomposition import PCA


def dnn_activation_deprecated(input, netname, layer, channel=None, column=None,
                              fe_axis=None, fe_meth=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    input[dataloader]: input image dataloader	
    netname[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)
    column[list]: column of interest
    fe_axis{str}: axis for feature extraction
    fe_meth[str]: feature extraction method, max, mean, median

    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 3D array with its shape as (n_picture, n_channel, n_column)
    """
    assert (fe_axis is None) == (fe_meth is None), 'Please specify fe_axis and fe_meth at the same time.'

    # get dnn activation
    loader = dio.NetLoader(netname)
    actmodel = dnn_truncate(loader, layer)
    actmodel.eval()
    dnnact = []
    count = 0  # count the progress
    for picdata, _ in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
        count += dnnact_part.shape[0]
        print('Extracted acts:', count)
    dnnact = np.array(dnnact)
    dnnact = dnnact.reshape((dnnact.shape[0], dnnact.shape[1], -1))

    # mask the data
    if channel is not None:
        dnnact = dnnact[:, channel, :]
    if column is not None: 
        dnnact = dnnact[:, :, column]
        
    # feature extraction
    if fe_axis is not None:
        fe_meths = {
            'max': np.max,
            'mean': np.mean,
            'median': np.median
        }

        if fe_axis == 'layer':
            dnnact = dnnact.reshape((dnnact.shape[0], -1))
            dnnact = fe_meths[fe_meth](dnnact, -1)[:, np.newaxis, np.newaxis]
        elif fe_axis == 'channel':
            dnnact = fe_meths[fe_meth](dnnact, 1)[:, np.newaxis, :]
        elif fe_axis == 'column':
            dnnact = fe_meths[fe_meth](dnnact, 2)[:, :, np.newaxis]
        else:
            raise ValueError('fe_axis should be layer, channel or column')
    
    return dnnact


def dnn_activation(data_loader, net_loader, layer):
    """
    Extract DNN activation from the specified layer

    Parameters:
    ------------
    data_loader[DataLoader]: stimuli loader
    net_loader[NetLoader]: neural network loader
    layer[str]: layer name of a DNN

    Returns:
    ---------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    raw_shape[tuple]: the DNN activation's raw shape
    """
    # change to eval mode
    net_loader.model.eval()

    # prepare dnn activation hook
    dnn_acts = []

    def hook_act(module, input, output):
        dnn_acts.extend(output.detach().numpy().copy())

    module = net_loader.model
    for k in net_loader.layer2keys[layer]:
        module = module._modules[k]
    hook_handle = module.register_forward_hook(hook_act)

    # extract dnn activation
    for stims, _ in data_loader:
        net_loader.model(stims)
        print('Extracted acts:', len(dnn_acts))
    dnn_acts = np.asarray(dnn_acts)
    raw_shape = dnn_acts.shape
    dnn_acts = dnn_acts.reshape((raw_shape[0], raw_shape[1], -1))

    hook_handle.remove()
    return dnn_acts, raw_shape


def dnn_mask(dnn_acts, chn=None, col=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    chn[list]: channel indices of interest
    col[list]: column indices of interest

    Returns:
    ---------
    dnn_acts[array]: DNN activation after mask
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    if chn is not None:
        dnn_acts = dnn_acts[:, chn, :]
    if col is not None:
        dnn_acts = dnn_acts[:, :, col]

    return dnn_acts


def dnn_pooling(dnn_acts, method):
    """
    Pooling DNN activation for each channel

    Parameters:
    ------------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    method[str]: pooling method, choices=(max, mean, median)

    Returns:
    ---------
    dnn_acts[array]: DNN activation after pooling
        a 3D array with its shape as (n_stim, n_chn, 1)
    """
    if method == 'max':
        dnn_acts = np.max(dnn_acts, 2)[:, :, None]
    elif method == 'mean':
        dnn_acts = np.mean(dnn_acts, 2)[:, :, None]
    elif method == 'median':
        dnn_acts = np.median(dnn_acts, 2)[:, :, None]
    else:
        raise ValueError('Not supported method:', method)

    return dnn_acts


def dnn_fe(dnn_acts, meth, n_feat, axis=None):
    """
    Extract features of DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    meth[str]: feature extraction method, choices=(pca, hist, psd)
        pca: use n_feat principal components as features
        hist: use histogram of activation as features
            Note: n_feat equal-width bins in the given range will be used!
        psd: use power spectral density as features
    n_feat[int]: The number of features to extract
    axis{str}: axis for feature extraction, choices=(chn, col)
        If it's None, extract features from the whole layer. Note:
        The result of this will be an array with shape (n_stim, n_feat, 1), but
        We also regard it as (n_stim, n_chn, n_col)

    Returns:
    -------
    dnn_acts_new[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    # adjust iterative axis
    n_stim, n_chn, n_col = dnn_acts.shape
    if axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
    elif axis == 'chn':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif axis == 'col':
        pass
    else:
        raise ValueError('not supported axis:', axis)
    _, n_iter, _ = dnn_acts.shape

    # extract features
    dnn_acts_new = np.zeros((n_stim, n_iter, n_feat))
    if meth == 'pca':
        pca = PCA(n_components=n_feat)
        for i in range(n_iter):
            dnn_acts_new[:, i, :] = pca.fit_transform(dnn_acts[:, i, :])
    elif meth == 'hist':
        for i in range(n_iter):
            for j in range(n_stim):
                dnn_acts_new[j, i, :] = np.histogram(dnn_acts[j, i, :], n_feat)[0]
    elif meth == 'psd':
        for i in range(n_iter):
            for j in range(n_stim):
                f, p = periodogram(dnn_acts[j, i, :])
                dnn_acts_new[j, i, :] = p[:n_feat]
    else:
        raise ValueError('not supported method:', meth)

    # adjust iterative axis
    if axis is None:
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))
    elif axis == 'chn':
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))

    return dnn_acts_new


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event, n_sample]
    onsets[array_like]: in sec. size = n_event
    durations[array_like]: in sec. size = n_event
    n_vol[int]: the number of volumes of BOLD signal
    tr[float]: repeat time in second
    ops[int]: oversampling number per second

    Returns:
    ---------
    X_hrfed[array]: the result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), 'The length of onsets and durations should ' \
                                                                       'be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        onsets = np.insert(onsets, 0, 0, 0)
        durations = np.insert(durations, 0, onset_min, 0)
        onset_min = 0
    elif onset_min < 0:
        print("The earliest event's onset is earlier than the start point of response.\n"
              "We clip the earlier time points after hrf_convolution to align with the response.")
        n_clipped = int(-onset_min * ops)

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  # compute volume acquisition timing
    n_time_point = int(((onsets + durations).max()-onset_min) * ops)
    X_hrfed = np.zeros([n_vol, 0])
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx+1]]
        # generate X raw time course
        X_tc = np.zeros((n_time_point, X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')
        X_tc_hrfed = X_tc_hrfed[n_clipped:, :]

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

    return X_hrfed
