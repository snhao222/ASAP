import torch
from torch import nn
import numpy as np
import pandas as pd
import csv
import yaml
#from torch2trt import torch2trt

class model(nn.Module):
    """Generate convolution-like models.
    
    Attributes:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        kernel_size (int): Kernel size.
        stride (int): Stride.
    """
    def __init__(self, type_m, in_channels=16, out_channels=32, kernel_size=3, stride=1):
        super(model, self).__init__()
        self.type_m = type_m
        if self.type_m == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        elif self.type_m == 'pool':
            self.pool = nn.MaxPool2d(kernel_size, stride)
    def forward(self, x):
        if self.type_m == 'conv':
            x = self.conv(x)
        elif self.type_m == 'pool':
            x = self.pool(x)
        return x

def warm_up(model):
    """Warm device up before measurement.
    
    Args:
        model (object): Class 'model'.
    """
    device = torch.device("cuda:0")
    print(device)
    model_0 = model('conv').eval().to(device)
    dummy_input = torch.randn(1, 16, 32, 32, dtype=torch.float).to(device)
    #model_0 = torch2trt(model_0, [dummy_input])
    for _ in range(300):
        model_0(dummy_input)
    print("Warm up completed.")

def measurement_core(model, type_m, repetitions, w, h, cin, cout, k, s):
    """Measure model inference latency several times and calculate the numerical average.
    
    Args:
        model (object): Class 'model'.
        repetitions (int): Measurement repetitions.
        w (int): Width.
        h (int): Height.
        cin (int): Input channel.
        cout (int): Output channel.
    
    Returns:
        An average inference latency.
    """
    device = torch.device("cuda:0")
    model_1 = model(type_m, in_channels=cin, out_channels=cout, kernel_size=k, stride=s).to(device).eval()
    dummy_input = torch.randn(1, cin, w, h, dtype=torch.float).to(device)
    #model_1 = torch2trt(model_1, [dummy_input])
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    for rep in range(repetitions):
        torch.cuda.synchronize()
        starter.record()
        model_1(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions

    return mean_syn

def para_measurement(model, w, h, cout, l_w, l_h, l_cout, k_conv, s_conv, k_pool, s_pool, frq):
    """Measure inference latency and calculate parameters for operator level predictor.
    
    Args:
        model (object): Class 'model'.
        w (int): Basic width.
        h (int): Basic height.
        cout (int): Basic output channel.
        l_w (int): Step length of width.
        l_h (int): Step length of height.
        l_cout (int): Step length of output channel.
        k_conv (list): A list of convolution kernels to be measured.
        s_conv (list): A list of convolution strides to be measured.
        k_pool (list): A list of pooling kernels to be measured.
        s_pool (list): A list of pooling strides to be measured.
        frq (int): Computing frequency of processor.
    """
    repetitions = 500
    latency_conv = []
    latency_pool = []
    latency_w = []
    latency_h = []
    latency_co = []
    latency_k_conv = []
    latency_s_conv = []
    latency_k_pool = []
    latency_s_pool = []
    for _ in range(len(k_conv[0])):
        latency_k_conv.append([])
    for _ in range(len(s_conv[0])):
        latency_s_conv.append([])
    for _ in range(len(k_pool[0])):
        latency_k_pool.append([])
    for _ in range(len(s_pool[0])):
        latency_s_pool.append([])

    cin_id = []
    with torch.no_grad():
        with open("./parameter_measurement.csv", "wt", newline='') as csvfile:
            fieldnames = ["c_in", "latency_conv_base", "latency_w", "latency_h", "latency_co", 'latency_pool_base']
            writer = csv.DictWriter(csvfile, fieldnames)
            writer.writeheader()
            for i in range(256):
                cin = 3 + i
                latency_conv_t = measurement_core(model, 'conv', repetitions, w, h, cin, cout, k_conv[1], s_conv[1])
                latency_w_t = measurement_core(model, 'conv', repetitions, w + l_w, h, cin, cout, k_conv[1], s_conv[1])
                latency_conv.append(latency_conv_t)
                latency_w.append(latency_w_t)
                latency_h_t = measurement_core(model, 'conv', repetitions, w, h + l_h, cin, cout, k_conv[1], s_conv[1])
                latency_h.append(latency_h_t)
                latency_co_t = measurement_core(model, 'conv', repetitions, w, h, cin, cout + l_cout, k_conv[1], s_conv[1])
                latency_co.append(latency_co_t)
                cin_id.append(cin)

                latency_pool_t = measurement_core(model, 'pool', repetitions, w, h, cin, cout, k_pool[1], s_pool[1])
                latency_pool.append(latency_pool_t)

                print("Round ", i)
                writer.writerow(
                                {
                                    "c_in": cin,
                                    'latency_conv_base': latency_conv_t,
                                    'latency_w': latency_w_t,
                                    'latency_h': latency_h_t,
                                    'latency_co': latency_co_t,
                                    'latency_pool_base': latency_pool_t
                                }
                            )

                for i, k_c in enumerate(k_conv[0]):
                    latency_k = measurement_core(model, 'conv', repetitions, w, h, cin, cout, k_c, s_conv[1])
                    latency_k_conv[i].append(latency_k)
                    torch.cuda.empty_cache()
                for i, s_c in enumerate(s_conv[0]):
                    latency_s = measurement_core(model, 'conv', repetitions, w, h, cin, cout, k_conv[1], s_c)
                    latency_s_conv[i].append(latency_s)
                    torch.cuda.empty_cache()
                for i, k_p in enumerate(k_pool[0]):
                    latency_k = measurement_core(model, 'pool', repetitions, w, h, cin, cout, k_p, s_pool[1])
                    latency_k_pool[i].append(latency_k)
                    torch.cuda.empty_cache()
                for i, s_p in enumerate(s_pool[0]):
                    latency_s = measurement_core(model, 'pool', repetitions, w, h, cin, cout, k_pool[1], s_p)
                    latency_s_pool[i].append(latency_s)
                    torch.cuda.empty_cache()

            k_w = float(np.average(np.divide(latency_w, latency_conv)))
            k_h = float(np.average(np.divide(latency_h, latency_conv)))
            k_co = float(np.average(np.divide(latency_co, latency_conv)))
            print("k_w:", k_w)
            print("k_h:", k_h)
            print("k_co:", k_co)

            ratio_k_conv_lst = []
            for i in range(len(latency_k_conv)):
                ratio_k_conv = float(np.average(np.divide(latency_k_conv[i], latency_conv)))
                print("conv_k = {}: {}" .format(k_conv[0][i], ratio_k_conv))
                ratio_k_conv_lst.append(ratio_k_conv)

            ratio_s_conv_lst = []
            for i in range(len(latency_s_conv)):
                ratio_s_conv = float(np.average(np.divide(latency_s_conv[i], latency_conv)))
                print("conv_s = {}: {}" .format(s_conv[0][i], ratio_s_conv))
                ratio_s_conv_lst.append(ratio_s_conv)

            ratio_k_pool_lst = []
            for i in range(len(latency_k_pool)):
                ratio_k_pool = float(np.average(np.divide(latency_k_pool[i], latency_pool)))
                print("pool_k = {}: {}" .format(k_pool[0][i], ratio_k_pool))
                ratio_k_pool_lst.append(ratio_k_pool)

            ratio_s_pool_lst = []
            for i in range(len(latency_s_pool)):
                ratio_s_pool = float(np.average(np.divide(latency_s_pool[i], latency_pool)))
                print("pool_s = {}: {}" .format(s_pool[0][i], ratio_s_pool))
                ratio_s_pool_lst.append(ratio_s_pool)

        ydata = {
                "w_0": w_0,
                "h_0": h_0,
                "cout_0": cout_0,
                "l_w": l_w,
                "l_h": l_h,
                "l_cout": l_cout,
                "k_w": k_w,
                "k_h": k_h,
                "k_co": k_co,
                'k_conv': k_conv[0],
                'k_conv_0': k_conv[1],
                's_conv': s_conv[0],
                's_conv_0': s_conv[1],
                'k_pool': k_pool[0],
                'k_pool_0': k_pool[1],
                's_pool': s_pool[0],
                's_pool_0': s_pool[1],
                "r_k_conv": ratio_k_conv_lst,
                "r_s_conv": ratio_s_conv_lst,
                "r_k_pool": ratio_k_pool_lst,
                "r_s_pool": ratio_s_pool_lst,
                "frq": frq
            }
        print(ydata)
        with open('para.yml', 'w', encoding='utf-8') as f:
            yaml.dump(data=ydata, stream=f, allow_unicode=True)

# Basic configurations of convolution operator.
w_0 = 116
h_0 = 120
cout_0 = 48
k_conv_0 = 3
s_conv_0 = 1
# Step lengths of convolution operator.
l_w = 8
l_h = 16
l_cout = 32
# Basic configurations of pooling operator.
k_pool_0 = 2
s_pool_0 = 2
# Kernel and stride configurations of convolution and pooling to be measured.
k_conv = [[1, 2, 4, 5, 7, 11], k_conv_0]
s_conv = [[2, 3, 4, 5], s_conv_0]
k_pool = [[1, 3, 4, 5], k_pool_0]
s_pool = [[1, 3, 4, 5], s_pool_0]
# Computing frequency of processor.
frq = 691200

warm_up(model)
para_measurement(model, w_0, h_0, cout_0, l_w, l_h, l_cout, k_conv, s_conv, k_pool, s_pool, frq)
