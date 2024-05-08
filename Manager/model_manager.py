__package__ = 'Model'
import torch
from torch import nn
from torchvision.models import vgg16, resnet18, alexnet
import copy
import math
from functools import reduce
from .resnet import resnet34, resnet50, resnet101, resnet152
from .util import decom_vgg16
from .darknet import Darknet19
from .testnet import TestModel

class Model_Manager():
    """Model generation and analysis.
    
    Attributes:
    task_type (str): Model name of task.
    nm_lst (list): A list contains operator names of model.
    type_lst (list): A list contains operator types of model.
    para_lst (list): A list contains operator parameter information of model.
    block_lst (list): A list contains model structure.
    t_nm_lst (list): A list contains operator names of test model.
    t_type_lst (list): A list contains operator types of test model.
    t_para_lst (list): A list contains operator parameter information of test model.
    t_block_lst (list): A list contains structure of test model.
    """
    def __init__(self, task_type):

        self.task_type = task_type
        if task_type == 'vgg':
            self.model_0 = vgg16(pretrained=True).features
        elif task_type == 'faster-rcnn':
            self.model_0 = decom_vgg16(True)
            pretext_model = torch.load('./model_data/voc_weights_vgg.pth', map_location='cpu')
            model_dict = self.model_0.state_dict()
            state_dict = {k:v for k,v in pretext_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model_0.load_state_dict(model_dict)
            self.model_0 = self.model_0.extractor
        elif task_type == 'resnet34':
            self.model_0 = resnet34(True)

        elif task_type == 'resnet50':
            self.model_0 = resnet50(True)
        
        elif task_type == 'resnet101':
            self.model_0 = resnet101(True)
        
        elif task_type == 'resnet152':
            self.model_0 = resnet152(True)
        
        elif task_type == 'darknet':
            self.model_0 = Darknet19(num_classes=1000, pretrained=True).features

        elif task_type == 'alexnet':
            self.model_0 = alexnet(pretrained=True).features

        self.nm_lst, self.type_lst, self.para_lst, self.block_lst = self.para_extractor(self.model_0)
        self.test_model = TestModel().features
        self.t_nm_lst, self.t_type_lst, self.t_para_lst, self.t_block_lst = self.para_extractor(self.test_model)
    
    def model_gen(self, nm_lst):
        """Generate models.

        Support generating partial models.
        
        Args:
            nm_lst (list): A list of name contains several operators in the model to be generated.
        
        Returns:
            A generated model.
        """
        model = copy.deepcopy(self.model_0).eval()
        for name, module in list(model.named_modules()):
            if not name in nm_lst:
                if name == '':
                    continue
                flag = False
                for nm in self.nm_lst:
                    if ((name+'.') in nm) and (len(name) < len(nm)) and (name[0] == nm[0]):
                        flag = True
                        break
                if flag:
                    continue

                if '.' in name:
                    b = ''
                    for content in name.split('.'):
                        if content.isdigit():
                            b += ('['+content+']')
                        else:
                            b += '.'+ content
                    name = b[1:]

                try:
                    exec('model.'+name + ' = nn.Sequential()')
                except:
                    try:
                        exec('model['+name + ' = nn.Sequential()')
                    except:
                        exec('model['+name + '] = nn.Sequential()')
        return model

    def para_extractor(self, model):
        """Extract operator parameter information from model.
        
        Args:
            model (object): Target model.
        
        Returns:
            A tuple contains nm_lst, type_lst, para_lst, and block_lst.
        """
        nm_lst = []
        type_lst = []
        para_lst = []
        block_lst = []
        for name, module in model.named_modules():

            if isinstance(module, nn.Conv2d):
                if 'extractor.' in name:
                    name = name.split('.')[-1]
                nm_lst.append(name)
                type_lst.append("conv2d")
                para_lst.append({
                    'kernel': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding[1],
                    'out_channels': module.out_channels
                })
                if len(name)>3:
                    flag = True
                    for i, bl in enumerate(block_lst):
                        if name.split('.')[0]+'.'+name.split('.')[1] in bl[0]: 
                            block_lst[i].append(name)
                            flag = False
                            break
                    if flag:
                        block_lst.append([name])
                else:
                    block_lst.append([name])

            elif isinstance(module, nn.MaxPool2d):
                if 'extractor.' in name:
                    name = name.split('.')[-1]
                nm_lst.append(name)
                type_lst.append("maxpool2d")
                para_lst.append({
                    'kernel': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                })
                if len(name)>3:
                    flag = True
                    for i, bl in enumerate(block_lst):
                        if name.split('.')[0]+'.'+name.split('.')[1] in bl[0]:
                            block_lst[i].append(name)
                            flag = False
                            break
                    if flag:
                        block_lst.append([name])
                else:
                    block_lst.append([name])

            elif isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):
                if 'extractor.' in name:
                    name = name.split('.')[-1]
                nm_lst.append(name)
                type_lst.append("relu")
                para_lst.append(None)
                if len(name)>3:
                    flag = True
                    for i, bl in enumerate(block_lst):
                        if name.split('.')[0]+'.'+name.split('.')[1] in bl[0]:
                            block_lst[i].append(name)
                            flag = False
                            break
                    if flag:
                        block_lst.append([name])
                else:
                    block_lst.append([name])
            
            elif isinstance(module, nn.BatchNorm2d):
                if 'extractor.' in name:
                    name = name.split('.')[-1]
                nm_lst.append(name)
                type_lst.append("batchnorm2d")
                para_lst.append(None)
                if len(name)>3:
                    flag = True
                    for i, bl in enumerate(block_lst):
                        if name.split('.')[0]+'.'+name.split('.')[1] in bl[0]:
                            block_lst[i].append(name)
                            flag = False
                            break
                    if flag:
                        block_lst.append([name])
                else:
                    block_lst.append([name])
        return nm_lst, type_lst, para_lst, block_lst
    
    def out_deduction(self, range_in, nm_lst):
        """Calculate the output range corresponding to input range.
        
        Args:
            range_in (list): input range contains input channel, width, and height.
            nm_lst (list): A list contains operator names of model.

        Returns:
            range_lst (list): A list of output range contains output channel, width, and height.
            range_w (list): A list of output width.
        """
        range_lst = []
        range_w = []
        range_lst.append(range_in)
        range_w.append(range_in[1])
        width_out = range_in[1]
        height_out = range_in[2]
        cout = range_in[0]
        for nm in nm_lst:
            if 'downsample' in nm:
                if self.type_lst[self.nm_lst.index(nm)] == 'conv2d':
                    k = self.para_lst[self.nm_lst.index(nm)]['kernel'][0]
                    s = self.para_lst[self.nm_lst.index(nm)]['stride'][0]
                    p = self.para_lst[self.nm_lst.index(nm)]['padding']
                    for _name in nm_lst:
                        if nm[:4] in _name:
                            cout, width_out, height_out = range_lst[nm_lst.index(_name)]
                            break
                    width_out = math.floor((width_out + 2*p - k) / s) + 1
                    height_out = math.floor((height_out + 2*p - k) / s) + 1
                    cout = self.para_lst[self.nm_lst.index(nm)]['out_channels']
                    range_lst.append([cout, width_out, height_out])
                    range_w.append(width_out)
                else:
                    range_lst.append([cout, width_out, height_out])
                    range_w.append(width_out)
                continue

            if self.type_lst[self.nm_lst.index(nm)] == 'conv2d':
                k = self.para_lst[self.nm_lst.index(nm)]['kernel'][0]
                s = self.para_lst[self.nm_lst.index(nm)]['stride'][0]
                p = self.para_lst[self.nm_lst.index(nm)]['padding']
                width_out = math.floor((width_out + 2*p - k) / s) + 1
                height_out = math.floor((height_out + 2*p - k) / s) + 1
                cout = self.para_lst[self.nm_lst.index(nm)]['out_channels']
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)

            elif self.type_lst[self.nm_lst.index(nm)] == 'maxpool2d':
                k = self.para_lst[self.nm_lst.index(nm)]['kernel']
                s = self.para_lst[self.nm_lst.index(nm)]['stride']
                p = self.para_lst[self.nm_lst.index(nm)]['padding']
                width_out = math.floor((width_out + 2*p - k) / s) + 1
                height_out = math.floor((height_out + 2*p - k) / s) + 1
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)

            else:
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)
        return range_lst, range_w
    
    def MRT(self, range_out, p_slice, nm_lst, type_lst, para_lst, range_lst):
        """Mapping range transforming.
        
        Calculate the input range corresponding to output range.

        Args:
            range_out (list): The output range of model.
            nm_lst (list): A list contains operator names of model.
            type_lst (list): A list contains operator types of model.
            para_lst (list): A list contains operator parameter information of model.
            range_lst (list): A list of complete output width.
        
        Returns:
            Intermediate range and corresponding padding parameters for each layer.
        """
        padding_lst = []
        range_in = range_out
        range_in_lst = [range_out[1]-range_out[0]+1]
        for nm in nm_lst[::-1]:
            p_slice = [0, 0]
            len_out = range_in[1] - range_in[0] + 1
            if 'downsample' in nm:
                padding_lst.append([None])
                if type_lst[nm_lst.index(nm)] == 'conv2d':
                    k = para_lst[nm_lst.index(nm)]['kernel'][0]
                    s = para_lst[nm_lst.index(nm)]['stride'][0]
                    len_in = (len_out -1) * s + k
                    range_in_lst.append(len_in)
                elif type_lst[nm_lst.index(nm)] == 'batchnorm2d':
                    range_in_lst.append(len_out)
                continue
            if type_lst[nm_lst.index(nm)] == 'conv2d':
                k = para_lst[nm_lst.index(nm)]['kernel'][0]
                s = para_lst[nm_lst.index(nm)]['stride'][0]
                p = para_lst[nm_lst.index(nm)]['padding']
                len_in = (len_out -1) * s + k
                if (range_in[0] * s - p) < 0:
                    p_slice[0] = -(range_in[0] * s - p)
                
                if (range_in[0] * s - p + len_in) > range_lst[nm_lst.index(nm)]:
                    p_slice[1] = range_in[0]*s-p + len_in - range_lst[nm_lst.index(nm)]

                seq_s = max(0, range_in[0] * s - p)

                len_in = (len_out -1) * s + k - (p_slice[0] + p_slice[1])

                range_in = [seq_s, seq_s+len_in-1]
                range_in_lst.append(len_in)

                padding_lst.append([p_slice[0], p_slice[1]])

            elif type_lst[nm_lst.index(nm)] == 'maxpool2d':
                k = para_lst[nm_lst.index(nm)]['kernel']
                s = para_lst[nm_lst.index(nm)]['stride']
                p = para_lst[nm_lst.index(nm)]['padding']
                len_in = (len_out -1) * s + k
                if (range_in[0] * s - p) < 0:
                    p_slice[0] = -(range_in[0] * s - p)
                if (range_in[0] * s - p + len_in) > range_lst[nm_lst.index(nm)]:
                    p_slice[1] = range_in[0]*s-p + len_in - range_lst[nm_lst.index(nm)]
                seq_s = max(0, range_in[0] * s - p)

                len_in = (len_out -1) * s + k - (p_slice[0] + p_slice[1])
                
                range_in = [seq_s, seq_s+len_in-1]
                range_in_lst.append(len_in)

                padding_lst.append([p_slice[0], p_slice[1]])

            else:
                padding_lst.append([None])
                range_in_lst.append(len_out)
        return range_in, padding_lst[::-1], range_in_lst[::-1]

    def model_padding(self, model, nm_lst, padding_lst):
        """Perform model padding according to padding list.
        
        Args:
            model (object): Target model.
            nm_lst (list): A list contains operator names of model.
            padding_lst (list): A list of padding parameters for each layer.
        """
        nm_lst_ = []
        for nm in nm_lst:
            if 'extractor.' in nm:
                nm_lst_.append(nm.split('.')[-1])
            else:
                nm_lst_.append(nm)

        for i, name in enumerate(nm_lst):
            if self.type_lst[self.nm_lst.index(name)] == 'conv2d' or self.type_lst[self.nm_lst.index(name)] == 'maxpool2d':
                name = nm_lst_[i]
                if '.' in name:
                    b = ''
                    for content in name.split('.'):
                        if content.isdigit():
                            b += ('['+content+']')
                        else:
                            b += '.'+ content
                    m_name = b[1:]
                else:
                    m_name = name

                if not padding_lst[i] == [None]:
                    module = reduce(getattr, name.split(sep='.'), model)
                    if isinstance(module, nn.Conv2d):
                        padding = ((module.padding)[1], (module.padding)[1], padding_lst[nm_lst_.index(name)][0], padding_lst[nm_lst_.index(name)][1])
                        module.padding = (0,0)
                    elif isinstance(module, nn.MaxPool2d):
                        padding = (module.padding, module.padding, padding_lst[nm_lst_.index(name)][0], padding_lst[nm_lst_.index(name)][1])
                        module.padding = (0,0)
                    try:
                        exec('model.'+ m_name+ '= nn.Sequential(nn.ZeroPad2d(padding), module)')
                    except:
                        try:
                            exec('model['+ m_name+ '= nn.Sequential(nn.ZeroPad2d(padding), module)')
                        except:
                            exec('model['+ m_name+ ']= nn.Sequential(nn.ZeroPad2d(padding), module)')
    
    def out_deduction_test(self, range_in, nm_lst):
        """Calculate the output range corresponding to input range for the test model."""
        range_lst = []
        range_w = []
        range_lst.append(range_in)
        range_w.append(range_in[1])
        width_out = range_in[1]
        height_out = range_in[2]
        cout = range_in[0]
        for nm in nm_lst:
            if 'downsample' in nm:
                if self.t_type_lst[self.t_nm_lst.index(nm)] == 'conv2d':
                    k = self.t_para_lst[self.t_nm_lst.index(nm)]['kernel'][0]
                    s = self.t_para_lst[self.t_nm_lst.index(nm)]['stride'][0]
                    p = self.t_para_lst[self.t_nm_lst.index(nm)]['padding']
                    for _name in nm_lst:
                        if nm[:4] in _name:
                            cout, width_out, height_out = range_lst[nm_lst.index(_name)]
                            break
                    width_out = math.floor((width_out + 2*p - k) / s) + 1
                    height_out = math.floor((height_out + 2*p - k) / s) + 1
                    cout = self.t_para_lst[self.t_nm_lst.index(nm)]['out_channels']
                    range_lst.append([cout, width_out, height_out])
                    range_w.append(width_out)
                else:
                    range_lst.append([cout, width_out, height_out])
                    range_w.append(width_out)
                continue

            if self.t_type_lst[self.t_nm_lst.index(nm)] == 'conv2d':
                k = self.t_para_lst[self.t_nm_lst.index(nm)]['kernel'][0]
                s = self.t_para_lst[self.t_nm_lst.index(nm)]['stride'][0]
                p = self.t_para_lst[self.t_nm_lst.index(nm)]['padding']
                width_out = math.floor((width_out + 2*p - k) / s) + 1
                height_out = math.floor((height_out + 2*p - k) / s) + 1
                cout = self.t_para_lst[self.t_nm_lst.index(nm)]['out_channels']
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)

            elif self.t_type_lst[self.t_nm_lst.index(nm)] == 'maxpool2d':
                k = self.t_para_lst[self.t_nm_lst.index(nm)]['kernel']
                s = self.t_para_lst[self.t_nm_lst.index(nm)]['stride']
                p = self.t_para_lst[self.t_nm_lst.index(nm)]['padding']
                width_out = math.floor((width_out + 2*p - k) / s) + 1
                height_out = math.floor((height_out + 2*p - k) / s) + 1
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)

            else:
                range_lst.append([cout, width_out, height_out])
                range_w.append(width_out)
        return range_lst, range_w
    
    def model_padding_test(self, model, nm_lst, padding_lst):
        """Perform model padding for the test model."""
        nm_lst_ = []
        for nm in nm_lst:
            if 'extractor.' in nm:
                nm_lst_.append(nm.split('.')[-1])
            else:
                nm_lst_.append(nm)

        for i, name in enumerate(nm_lst):
            if self.t_type_lst[self.t_nm_lst.index(name)] == 'conv2d' or self.t_type_lst[self.t_nm_lst.index(name)] == 'maxpool2d':
                name = nm_lst_[i]
                if '.' in name:
                    b = ''
                    for content in name.split('.'):
                        if content.isdigit():
                            b += ('['+content+']')
                        else:
                            b += '.'+ content
                    m_name = b[1:]
                else:
                    m_name = name

                if not padding_lst[i] == [None]:
                    module = reduce(getattr, name.split(sep='.'), model)
                    if isinstance(module, nn.Conv2d):
                        padding = ((module.padding)[1], (module.padding)[1], padding_lst[nm_lst_.index(name)][0], padding_lst[nm_lst_.index(name)][1])
                        module.padding = (0,0)
                    elif isinstance(module, nn.MaxPool2d):
                        padding = (module.padding, module.padding, padding_lst[nm_lst_.index(name)][0], padding_lst[nm_lst_.index(name)][1])
                        module.padding = (0,0)
                    try:
                        exec('model.'+ m_name+ '= nn.Sequential(nn.ZeroPad2d(padding), module)')
                    except:
                        try:
                            exec('model['+ m_name+ '= nn.Sequential(nn.ZeroPad2d(padding), module)')
                        except:
                            exec('model['+ m_name+ ']= nn.Sequential(nn.ZeroPad2d(padding), module)')
