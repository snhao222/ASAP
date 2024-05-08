import torch
import torch.nn as nn
import math
import numpy as np
import pickle

class Latency_predictor():
    """Lightweight accurate DL inference performance predictor.
    
    Attributes:
        configs (dict): A dictionary contains predictor configuration parameters.
        fitting (dict): A dictionary contains parameters of basic latency pattern.
    """
    def __init__(self, configs, fitting):
        self.ops = ['conv2d', 'maxpool2d', 'relu', 'bn', ['conv2d', 'relu']]
        self.w_0 = configs['w_0']
        self.h_0 = configs['h_0']
        self.cout_0 = configs['cout_0']
        self.l_w = configs['l_w']
        self.l_h = configs['l_h']
        self.l_cout = configs['l_cout']
        k_w = configs['k_w']
        k_h = configs['k_h']
        k_co = configs['k_co']
  
        self.k_conv_0 = configs['k_conv_0']
        self.k_conv = configs['k_conv']
        self.s_conv_0 = configs['s_conv_0']
        self.s_conv = configs['s_conv']

        self.k_pool_0 = configs['k_pool_0']
        self.k_pool = configs['k_pool']
        self.s_pool_0 = configs['s_pool_0']
        self.s_pool = configs['s_pool']
        self.r_k_conv = configs['r_k_conv']
        self.r_s_conv = configs['r_s_conv']
        self.r_k_pool = configs['r_k_pool']
        self.r_s_pool = configs['r_s_pool']
        self.frq0 = configs['frq0']
        self.frq = configs['frq']

        self.op_lst = fitting['op_lst']
        self.k_lst = fitting['k_lst']
        self.b_lst = fitting['b_lst']

        self.s_a_co = (k_co-1)/(1+math.ceil(self.cout_0/self.l_cout-1)-(math.ceil(self.cout_0/self.l_cout-1)*k_co))
        self.s_a_w = (k_w-1)/(1+math.ceil(self.w_0/self.l_w-1)-(math.ceil(self.w_0/self.l_w-1)*k_w))
        self.s_a_h = (k_h-1)/(1+math.ceil(self.h_0/self.l_h-1)-(math.ceil(self.h_0/self.l_h-1)*k_h))

        self.rule_set = []

    def operator_predictor(self, input_shape, type_l, para_l):
        """Operator-level latency prediction.
        
        Args:
            input_shape (list): feature map shape of input data.
            type_l (str): Operator type.
            para_l (dict): Operator hyperparameters.
        """
        width_in = input_shape[1]
        height_in = input_shape[2]
        cin = input_shape[0]
        if type_l == 'conv2d':
            k = para_l['kernel'][0]
            s = para_l['stride'][0]
            p = para_l['padding']
            co = para_l['out_channels']
            k_fit = self.k_lst[self.op_lst.index('conv')]
            b_fit = self.b_lst[self.op_lst.index('conv')]
            f0 = k_fit * cin + b_fit
            T_co = (1+math.ceil(co/self.l_cout-1)*self.s_a_co)/(1+math.ceil(self.cout_0/self.l_cout-1)*self.s_a_co)
            T_w = (1+math.ceil(width_in/self.l_w-1)*self.s_a_w)/(1+math.ceil(self.w_0/self.l_w-1)*self.s_a_w)
            T_h = (1+math.ceil(height_in/self.l_h-1)*self.s_a_h)/(1+math.ceil(self.h_0/self.l_h-1)*self.s_a_h)
            if k != self.k_conv_0:
                coef_k = self.r_k_conv[self.k_conv.index(k)]
            else:
                coef_k = 1
            if s != self.s_conv_0:
                coef_s = self.r_s_conv[self.s_conv.index(s)]
            else:
                coef_s = 1
            latency = f0 * T_co * T_w * T_h * coef_k * coef_s
            width_out = math.floor((width_in + 2*p - k) / s) + 1
            height_out = math.floor((height_in + 2*p - k) / s) + 1

        elif type_l == 'maxpool2d':
            k = para_l['kernel']
            s = para_l['stride']
            p = para_l['padding']
            k_fit = self.k_lst[self.op_lst.index('pool')]
            b_fit = self.b_lst[self.op_lst.index('pool')]
            co = cin
            if k != self.k_pool_0:
                coef_k = self.r_k_pool[self.k_pool.index(k)]
            else:
                coef_k = 1
            if s != self.s_pool_0:
                coef_s = self.r_s_pool[self.s_pool.index(s)]
            else:
                coef_s = 1
            latency = k_fit * cin * width_in * height_in + b_fit
            latency = latency * coef_k * coef_s
            width_out = math.floor((width_in + 2*p - k) / s) + 1
            height_out = math.floor((height_in + 2*p - k) / s) + 1

        elif type_l == 'relu':
            width_out = width_in
            height_out = height_in
            co = cin
            k_fit = self.k_lst[self.op_lst.index('relu')]
            b_fit = self.b_lst[self.op_lst.index('relu')]
            latency = k_fit * cin * width_in * height_in + b_fit

        elif type_l == 'batchnorm2d':
            width_out = width_in
            height_out = height_in
            co = cin
            k_fit = self.k_lst[self.op_lst.index('bn')]
            b_fit = self.b_lst[self.op_lst.index('bn')]
            latency = k_fit * cin * width_in * height_in + b_fit

        return latency, [co, width_out, height_out]
    
    def block_predictor(self, model, scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst):
        """Model block inference performance prediction.
        
        Fine-tuning the operator-level latency.

        Args:
            model (object): Fusion fine-tuning model.
            scaler : Data normalization scale parameter.
            range_lst (list): A list of complete data range.
            range_in_lst (list): A list of desired data range.
            nm_lst (list): A list contains operator names of model.
            type_lst (list): A list contains operator types of model.
            para_lst (list): A list contains operator parameter information of model.
        
        Returns:
            Inference latency of the target model.
        """
        latency_sum = 0
        op_pair = []
        latency_pair = []
        ops = ['conv2d', 'maxpool2d', 'relu', 'batchnorm2d', ['conv2d', 'relu']]
        rule_set = [['conv2d', 'maxpool2d'],['conv2d', 'batchnorm2d'],['conv2d', 'relu'], ['maxpool2d', 'batchnorm2d'], ['batchnorm2d', 'relu'], [['conv2d','relu'], 'maxpool2d']]
        for nm in nm_lst:
            if type_lst[nm_lst.index(nm)] == 'conv2d':
                input_shape = [range_lst[nm_lst.index(nm)][0], range_in_lst[nm_lst.index(nm)], range_lst[nm_lst.index(nm)][2]]
                latency, _ = self.operator_predictor(input_shape, type_lst[nm_lst.index(nm)], para_lst[nm_lst.index(nm)])
    
            elif type_lst[nm_lst.index(nm)] == 'maxpool2d':
                input_shape = [range_lst[nm_lst.index(nm)][0], range_in_lst[nm_lst.index(nm)], range_lst[nm_lst.index(nm)][2]]
                latency, _ = self.operator_predictor(input_shape, type_lst[nm_lst.index(nm)], para_lst[nm_lst.index(nm)])
    
            elif type_lst[nm_lst.index(nm)] == 'relu':
                input_shape = [range_lst[nm_lst.index(nm)][0], range_in_lst[nm_lst.index(nm)], range_lst[nm_lst.index(nm)][2]]
                latency, _ = self.operator_predictor(input_shape, type_lst[nm_lst.index(nm)], para_lst[nm_lst.index(nm)])
    
            elif type_lst[nm_lst.index(nm)] == 'batchnorm2d':
                input_shape = [range_lst[nm_lst.index(nm)][0], range_in_lst[nm_lst.index(nm)], range_lst[nm_lst.index(nm)][2]]
                latency, _ = self.operator_predictor(input_shape, type_lst[nm_lst.index(nm)], para_lst[nm_lst.index(nm)])
    
            op_pair.append(type_lst[nm_lst.index(nm)])
            latency_pair.append(latency)

            if len(op_pair) == 2:
                if [op_pair[0], op_pair[1]] in rule_set:
                    embed1, embed2 = np.zeros(len(ops)), np.zeros(len(ops))
                    embed1[ops.index(op_pair[0])]=1
                    embed2[ops.index(op_pair[1])]=1
                    x = np.concatenate((embed1, embed2, [latency_pair[0]], [latency_pair[1]]), axis=0).astype(np.float32)
                    x = np.expand_dims(x, 0)
                    x = scaler.transform(x)
                    x = torch.from_numpy(x).float()
                    
                    with torch.no_grad():
                        fused_latency = model(x).item()
                    op_pair = [[op_pair[0], op_pair[1]]]
                    latency_pair = [fused_latency]
                else:
                    latency_sum += latency_pair.pop(0)
                    op_pair.pop(0)
            if nm == nm_lst[-1]:
                for latency_left in latency_pair:
                    latency_sum += latency_left 
        return latency_sum * self.frq0 / self.frq