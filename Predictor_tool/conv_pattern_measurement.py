import torch
from torch import nn
import numpy as np
import pandas as pd
#from torch2trt import torch2trt

class model(nn.Module):
    """Generate convolution models.
    
    Attributes:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
    """
    def __init__(self, in_channels=16, out_channels=32):
        super(model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

def warm_up(model):
    """Warm device up before measurement.
    
    Args:
        model (object): Class 'model'.
    """
    device = torch.device("cuda:0")
    print(device)
    model_0 = model().eval().to(device)
    dummy_input = torch.randn(1, 16, 32, 32, dtype=torch.float).to(device)
    #model_0 = torch2trt(model_0, [dummy_input])
    for _ in range(300):
        model_0(dummy_input)
    print("Warm up completed.")

def measurement_core(model, repetitions, w, h, cin, cout):
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
    model_1 = model(in_channels=cin, out_channels=cout).to(device).eval()
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

def pattern_measurement(model):
    """Measure the latency pattern with different values of widths, heights, and output channels.
    
    Args:
        model (object): Class 'model'.
        repetitions (int): Measurement repetitions.
        latency_w (list): A list of inference latency under different widths.
        latency_h (list): A list of inference latency under different heights.
        latency_co (list): A list of inference latency under different output channels.
        w_id (int): Width.
        h_id (int): Height.
        cout_id (int): Output channel.
        cin (int): Input channel.
    """
    repetitions = 150
    latency_w = []
    latency_h = []
    latency_co = []
    w_id = []
    h_id = []
    cout_id = []
    cin = 16
    with open('conv_pattern_measurement.txt','w',encoding='utf-8') as f:
        f.write('W:'+'\n')
        for i in range(256):
            w = 3 + i
            h = 224
            cout = 16
            la_w = measurement_core(model, repetitions, w, h, cin, cout)
            latency_w.append(la_w)
            w_id.append(w)
            print(i)
            f.write(str(la_w)+'\n')
        print("w pattern measurement completed.")
        f.write('H:'+'\n')
        for i in range(256):
            w = 224
            h = 3 + i
            cout = 16
            la_h = measurement_core(model, repetitions, w, h, cin, cout)
            latency_h.append(la_h)
            h_id.append(h)
            print(i)
            f.write(str(la_h)+'\n')
        print("h pattern measurement completed.")
        f.write('Cout:'+'\n')
        for i in range(256):
            w = 224
            h = 224
            cout = 3 + i
            la_co = measurement_core(model, repetitions, w, h, cin, cout)
            latency_co.append(la_co)
            cout_id.append(cout)
            print(i)
            f.write(str(la_co)+'\n')
        print("cout pattern measurement completed.")
    dfData = {
        'w': w_id,
        'latency_w': latency_w,
        'h': h_id,
        'latency_h': latency_h,
        'co': cout_id,
        'latency_co': latency_co
    }
    df = pd.DataFrame(dfData)
    df.to_csv('conv_pattern_measurement.csv', index=False)

warm_up(model)
pattern_measurement(model)