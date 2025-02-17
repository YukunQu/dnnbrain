import os
import cv2
import scipy.io
import numpy as np

from torchvision import transforms
from collections import OrderedDict
from dnnbrain.dnn.models import Vgg_face

try:
    from PIL import Image
except ModuleNotFoundError:
    raise Exception('Please install pillow in your work station')

try:
    import torch
    import torchvision
except ModuleNotFoundError:
    raise Exception(
            'Please install pytorch and torchvision in your work station')

DNNBRAIN_MODEL_DIR = os.environ['DNNBRAIN_MODEL_DIR']


class PicDataset:
    """
    Build a dataset to load pictures
    """
    def __init__(self, par_path, pic_ids, conditions=None, transform=None, crops=None):
        """
        Initialize PicDataset

        Parameters:
        ------------
        par_path[str]: picture parent path
        pic_ids[sequence]: Each pic_id is a path which can find the picture file relative to par_path.
        conditions[sequence]: Each picture's condition.
        transform[callable function]: optional transform to be applied on a sample.
        crops[array]: 2D array with shape (n_pic, 4)
            Row index is corresponding to the index in pic_ids.
            Each row is a bounding box which is used to crop the picture.
            Each bounding box's four elements are:
                left_coord, upper_coord, right_coord, lower_coord.
        """
        self.par_path = par_path
        self.pic_ids = pic_ids
        self.conditions = np.ones(len(self.pic_ids)) if conditions is None else conditions
        self.conditions_uniq = np.unique(self.conditions).tolist()
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.crops = crops

    def __len__(self):
        """
        Return sample size
        """
        return len(self.pic_ids)

    def __getitem__(self, idx):
        """
        Get picture data and target label of each sample

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        pic_img: picture data, save as a pillow instance
        trg_label: target of each sample (label)
        """
        # load picture
        pic_img = Image.open(os.path.join(self.par_path, self.pic_ids[idx])).convert('RGB')

        # crop picture
        if self.crops is not None:
            pic_img = pic_img.crop(self.crops[idx])

        pic_img = self.transform(pic_img)  # transform picture
        trg_label = self.conditions_uniq.index(self.conditions[idx])  # get target
        return pic_img, trg_label

    def get_picname(self, idx):
        """
        Get picture name and its condition (target condition)

        Parameters:
        -----------
        idx: index of sample

        Returns:
        ---------
        picname: picture name
        condition: target condition
        """
        return os.path.basename(self.pic_ids[idx]), self.conditions[idx]


class VidDataset:
    """
    Dataset for video data
    """
    def __init__(self, vid_file, frame_nums, conditions=None, transform=None, crops=None):
        """
        Parameters:
        -----------
        vid_file[str]: video data file
        frame_nums[sequence]: sequence numbers of the frames of interest
        conditions[sequence]: each frame's condition
        transform[pytorch transform]
        crops[array]: 2D array with shape (n_pic, 4)
            Row index is corresponding to the index in frame_nums.
            Each row is a bounding box which is used to crop the frame.
            Each bounding box's four elements are:
                left_coord, upper_coord, right_coord, lower_coord.
        """
        self.vid_cap = cv2.VideoCapture(vid_file)
        self.frame_nums = frame_nums
        self.conditions = np.ones(len(self.frame_nums)) if conditions is None else conditions
        self.conditions_uniq = np.unique(self.conditions).tolist()
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.crops = crops

    def __getitem__(self, idx):
        # get frame
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_nums[idx]-1)
        _, frame = self.vid_cap.read()
        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # crop frame
        if self.crops is not None:
            frame_img = frame_img.crop(self.crops[idx])

        frame = self.transform(frame_img)  # transform frame
        trg_label = self.conditions_uniq.index(self.conditions[idx])  # get target
        return frame, trg_label

    def __len__(self):
        return len(self.frame_nums)


def read_imagefolder(parpath):
    """
    The function read from a already organized Image folder or a folder that only have pictures
    and return picpath list and condition list
    for generate csv file more quickly.

    Parameters:
    ----------
    parpath[str]: parent path of pictures
    
    Return:
    ------
    picpath[list]: contains all subpath of pictures in parpath
    condition[list]: contains categories of all pictures
    """
    test_set = list(os.walk(parpath))

    picpath = []
    condition = []
    if len(test_set) == 1:  # the folder only have pictures, the folder name will be the condition
        label = test_set[0]
        condition_name = os.path.basename(label[0])
        picpath_tem = label[2]
        condition_tem = [condition_name for i in label[2]]
        picpath.append(picpath_tem)
        condition.append(condition_tem)
    else:                   # the folder have have some sub-folders as pytorch ImageFolder,
        for label in test_set[1:]:
            condition_name = os.path.basename(label[0])
            picpath_tem = [condition_name + '/' + pic for pic in label[2]]
            condition_tem = [condition_name for i in label[2]]  # the sub-folders name will be the conditions.
            picpath.append(picpath_tem)
            condition.append(condition_tem)

    picpath = sum(picpath, [])
    condition = sum(condition, [])
    return picpath, condition


def save_activation(activation, outpath):
    """
    Save activaiton data as a csv file or mat format file to outpath
         csv format save a 2D.
            The first column is stimulus indexs
            The second column is channel indexs
            Each row is the activation of a filter for a picture
         mat format save a 2D or 4D array depend on the activation from
             convolution layer or fully connected layer.
            4D array Dimension:sitmulus x channel x pixel x pixel
            2D array Dimension:stimulus x activation
    Parameters:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    """
    imgname = os.path.basename(outpath)
    imgsuffix = imgname.split('.')[-1]

    if imgsuffix == 'csv':
        if len(activation.shape) == 4:
            activation2d = np.reshape(
                    activation, (np.prod(activation.shape[0:2]), -1,),
                    order='C')
            channelline = np.array(
                    [channel + 1 for channel
                     in range(activation.shape[1])] * activation.shape[0])
            stimline = []
            for i in range(activation.shape[0]):
                a = [i + 1 for j in range(activation.shape[1])]
                stimline = stimline + a
            stimline = np.array(stimline)
            channelline = np.reshape(channelline, (channelline.shape[0], 1))
            stimline = np.reshape(stimline, (stimline.shape[0], 1))
            activation2d = np.concatenate(
                    (stimline, channelline, activation2d), axis=1)
        elif len(activation.shape) == 2:
            stim_indexs = np.arange(1, activation.shape[0] + 1)
            stim_indexs = np.reshape(stim_indexs, (-1, stim_indexs[0]))
            activation2d = np.concatenate((stim_indexs, activation), axis=1)
        np.savetxt(outpath, activation2d, delimiter=',')
    elif imgsuffix == 'mat':
        scipy.io.savemat(outpath, mdict={'activation': activation})
    else:
        np.save(outpath, activation)


class NetLoader:
    def __init__(self, net=None):
        """
        Load neural network model

        Parameters:
        -----------
        net[str]: a neural network's name
        """
        netlist = ['alexnet', 'vgg11', 'vggface']
        if net in netlist:
            if net == 'alexnet':
                self.model = torchvision.models.alexnet()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'alexnet_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv1_relu': (0, 1), 'conv1_maxpool': (0, 2), 'conv2': (0, 3),
                                      'conv2_relu': (0, 4), 'conv2_maxpool': (0, 5), 'conv3': (0, 6), 'conv3_relu': (0, 7),
                                      'conv4': (0, 8), 'conv4_relu': (0, 9),'conv5': (0, 10), 'conv5_relu': (0, 11),
                                      'conv5_maxpool': (0, 12), 'fc1': (2, 1), 'fc1_relu': (2, 2),
                                      'fc2': (2, 4), 'fc2_relu': (2, 5), 'fc3': (2, 6), 'prefc': (2,)}
                self.layer2keys = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                                    'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                                    'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                                    'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                                    'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                                    'conv5': ('features', '10'), 'conv5_relu': ('features', '11'),
                                    'conv5_maxpool': ('features', '12'), 'fc1': ('classifier', '1'),
                                    'fc1_relu': ('classifier', '2'), 'fc2': ('classifier', '4'),
                                    'fc2_relu': ('classifier', '5'), 'fc3': ('classifier', '6')}
                self.img_size = (224, 224)
            elif net == 'vgg11':
                self.model = torchvision.models.vgg11()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg11_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv2': (0, 3),
                                      'conv3': (0, 6), 'conv4': (0, 8),
                                      'conv5': (0, 11), 'conv6': (0, 13),
                                      'conv7': (0, 16), 'conv8': (0, 18),
                                      'fc1': (2, 0), 'fc2': (2, 3),
                                      'fc3': (2, 6), 'prefc':(2,)}
                self.img_size = (224, 224)
            elif net == 'vggface':
                self.model = Vgg_face()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg_face_dag.pth')))
                self.layer2indices = {'conv1': (0,), 'conv2': (2,),
                                      'conv3': (5,), 'conv4': (7,),
                                      'conv5': (10,), 'conv6': (12,),
                                      'conv7': (14,), 'conv8': (17,),
                                      'conv9': (19,), 'conv10': (21,),
                                      'conv11': (24,), 'conv12': (26,),
                                      'conv13': (28,), 'fc1': (31,),
                                      'fc2': (34,), 'fc3': (37,), 'prefc':(31,)}
                self.img_size = (224, 224)
        else:
            print('Not internal supported, please call netloader function'
                  'to assign model, layer2indices and picture size.')
            self.model = None
            self.layer2indices = None
            self.img_size = None

    def load_model(self, dnn_model, model_param=None,
                   layer2indices=None, input_imgsize=None):
        """
        Load DNN model

        Parameters:
        -----------
        dnn_model[nn.Modules]: DNN model
        model_param[string/state_dict]: Parameters of DNN model
        layer2indices[dict]: Comparison table between layer name and
            DNN frame layer.
            Please make dictionary as following format:
                {'conv1': (0, 0), 'conv2': (0, 3), 'fc1': (2, 0)}
        input_imgsize[tuple]: the input picture size
        """
        self.model = dnn_model
        if model_param is not None:
            if isinstance(model_param, str):
                self.model.load_state_dict(torch.load(model_param))
            else:
                self.model.load_state_dict(model_param)
        self.layer2indices = layer2indices
        self.img_size = input_imgsize
        print('You had assigned a model into netloader.')


def read_dnn_csv(dnn_csv):
    """
    Read pre-designed dnn csv file.

    Parameters:
    -----------
    dnn_csv[str]: Path of csv file.
        Note that the suffix of dnn_csv is .db.csv.
        Format of db.csv of picture stimuli is
        --------------------------
        type:stimulus
        title:picture stimuli
        stimPath:parent_dir_to_pictures
        stimType:picture
        [Several optional keys] (eg., crop:True)
        variableName:stimID,[onset],[duration],[condition]
        pic1_path,0,1,cat
        pic2_path,1,1,dog
        pic3_path,2,1,cat
        ...,...,...,...

        Format of db.csv of video stimuli is
        --------------------------
        type:stimulus
        title:video stimuli
        stimPath:path_to_video_file
        stimType:video
        [Several optional keys] (eg., hrf_tr:2)
        variableName:stimID,[onset],[duration],[condition]
        1,0,1,cat
        2,1,1,dog
        3,2,1,cat
        ...,...,...,...

        Format of db.csv of response is
        --------------------------
        type:response
        title:visual roi
        [Several optional keys] (eg., tr:2)
        variableName:OFA,FFA
        123,312
        222,331
        342,341
        ...,...

        Format of dnn_csv of dmask is
        --------------------------
        type:dmask
        title:alexnet roi
        [Several optional keys]
        variableName:chn,col
        1,2,3,5,7,124,...
        3,4,...

    Return:
    -------
    dbcsv[dict]: Dictionary of the output variable
    """
    # ---Load csv data---
    assert '.db.csv' in dnn_csv, 'Suffix of dnn_csv should be .db.csv'
    with open(dnn_csv, 'r') as f:
        csv_data = f.read().splitlines()
    # remove null line
    while '' in csv_data:
        csv_data.remove('')
    meta_idx = ['variableName' in i for i in csv_data].index(True)
    csv_meta = csv_data[:meta_idx]
    csv_val = csv_data[meta_idx:]

    # ---Handle csv data---
    dbcsv = {}
    for cm in csv_meta:
        k, v = cm.split(':')
        dbcsv[k] = v
    assert 'type' in dbcsv.keys(), 'type needs to be included in csvfiles.'
    assert 'title' in dbcsv.keys(), 'title needs to be included in csvfiles.'

    # identify the type of data
    assert dbcsv['type'] in ['stimulus', 'dmask', 'response'], \
        'Type must be named as stimulus, dmask or response.'

    # Operate csv_val
    variable_keys = csv_val[0].split(':')[1].split(',')

    # if dmask, variableAxis is row, each row can have different length.
    if dbcsv['type'] == 'dmask':
        variable_data = [np.asarray(i.split(','), dtype=np.int) - 1 for i in csv_val[1:]]
    # if stim/resp, variableAxis is col, each col must have the same length.
    else:
        variable_data = [i.split(',') for i in csv_val[1:]]
        variable_data = list(zip(*variable_data))
        if dbcsv['type'] == 'stimulus':
            if dbcsv['stimType'] == 'picture':
                # data type for stimID or condition is str, others float.
                for i, v_i in enumerate(variable_data):
                    if variable_keys[i] in ['stimID', 'condition']:
                        variable_data[i] = np.asarray(v_i, dtype=np.str)
                    else:
                        variable_data[i] = np.asarray(v_i, dtype=np.float)
            elif dbcsv['stimType'] == 'video':
                for i, v in enumerate(variable_data):
                    if variable_keys[i] == 'stimID':
                        variable_data[i] = np.array(v, dtype=np.int)
                    elif variable_keys[i] == 'condition':
                        variable_data[i] = np.array(v, dtype=np.str)
                    else:
                        variable_data[i] = np.array(v, dtype=np.float)
            else:
                raise ValueError('not supported stimulus type: {}'.format(dbcsv['stimType']))
        elif dbcsv['type'] == 'response':
            variable_data = np.asarray(variable_data, dtype=np.float)
        else:
            raise ValueError('not supported csv type: {}'.format(dbcsv['type']))

    var_dict = OrderedDict()
    for idx, key in enumerate(variable_keys):
        var_dict[key] = variable_data[idx]
    dbcsv['var'] = var_dict
    return dbcsv


def save_dnn_csv(fpath, ftype, title, variables, opt_meta=None):
    """
    Generate dnn brain csv.

    Parameters:
    ------------
    fpath[str]: output file path, ending with .db.csv
    ftype[str]: file type, ['stimulus', 'dmask', 'response'].
    title[str]: customized title
    variables[dict]: dictionary of signals or data
    opt_meta[dict]: some other optional meta data
    """
    assert fpath.endswith('.db.csv'), "Suffix of dnnbrain csv file should be .db.csv"
    with open(fpath, 'w') as f:
        # First line, type
        f.write('type:{}\n'.format(ftype))
        # Second line, title
        f.write('title:{}\n'.format(title))
        # Optional meta data
        if opt_meta is not None:
            for k, v in opt_meta.items():
                f.write('{0}:{1}\n'.format(k, v))
        # variableName line
        f.write('variableName:{}\n'.format(','.join(variables.keys())))
        variable_vals = []
        if ftype == 'dmask':
            for variable_val in variables.values():
                variable_vals.append(','.join(map(str, variable_val)))
        else:
            variable_vals = np.array(list(variables.values()), dtype=np.str).T
            variable_vals = [','.join(row) for row in variable_vals]
        f.write('\n'.join(variable_vals))


def read_dmask_csv(fpath):
    """
    Read pre-designed .dmask.csv file.

    Parameters:
    ----------
    fpath: path of .dmask.csv file

    Return:
    ------
    dmask_dict[OrderedDict]: Dictionary of the DNN mask information
    """
    # -load csv data-
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath) as rf:
        lines = rf.read().splitlines()

    # extract layers, channels and columns of interest
    dmask_dict = OrderedDict()
    for l_idx, line in enumerate(lines):
        if '=' in line:
            # layer
            layer, axes = line.split('=')
            dmask_dict[layer] = {'chn': None, 'col': None}

            # channels and columns
            axes = axes.split(',')
            while '' in axes:
                axes.remove('')
            assert len(axes) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for a_idx, axis in enumerate(axes, 1):
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                numbers = [int(num) for num in lines[l_idx+a_idx].split(',')]
                dmask_dict[layer][axis] = numbers

    return dmask_dict


def save_dmask_csv(fpath, dmask_dict):
    """
    Generate .dmask.csv

    Parameters
    ---------
    fpath[str]: output file path, ending with .dmask.csv
    dmask_dict[dict]: Dictionary of the DNN mask information
    """
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath, 'w') as wf:
        for layer, axes_dict in dmask_dict.items():
            axes = []
            num_lines = []
            assert len(axes_dict) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for axis, numbers in axes_dict.items():
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                if numbers is not None:
                    axes.append(axis)
                    num_line = ','.join(map(str, numbers))
                    num_lines.append(num_line)

            wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
            for num_line in num_lines:
                wf.write(num_line+'\n')
