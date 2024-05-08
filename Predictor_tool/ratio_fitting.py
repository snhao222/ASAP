import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import leastsq
import csv
from tqdm import tqdm, trange
from thop import profile
#from torch2trt import torch2trt

class model(nn.Module):
    """Generate models with operators to be measured.
    
    Attributes:
        operators (list): A list of operator types.
        configrations (list): A list of operator configurations.
    """
    def __init__(self, operators, configrations, init_weights=True):
        super(model, self).__init__()
        self.model=nn.Sequential()
        i = 0
        for op in operators:
            if op == 0:
                self.model.add_module('{0}_conv'.format(i), nn.Conv2d(in_channels = configrations[i][0], out_channels = configrations[i][1], kernel_size=configrations[i][2], stride=configrations[i][3], padding=1))
            elif op == 1:
                self.model.add_module('{0}_pool'.format(i), nn.MaxPool2d(kernel_size = configrations[i][0], stride = configrations[i][1]))
            elif op == 2:
                self.model.add_module('{0}_bn'.format(i), nn.BatchNorm2d(num_features = configrations[i][0], affine=True))
            elif op == 3:
                self.model.add_module('{0}_relu'.format(i), nn.ReLU(inplace=False))
            i += 1
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

device = torch.device('cuda')

def warm_up(model):
    """Warm device up before measurement.
    
    Args:
        model (object): Class 'model'.
    """
    device = torch.device("cuda:0")
    print(device)
    model_0 = model([0], [[16, 16, 3, 1]]).eval().to(device)
    dummy_input = torch.randn(1, 16, 32, 32, dtype=torch.float).to(device)
    for _ in range(300):
        model_0(dummy_input)
    print("Warm up completed.")

def config_gen(op_pairs, num):
    """Generate random model configurations.

    Args:
        op_pairs (list): operator type.
        num (int): Generated data set quantity.
    """
    if op_pairs[0] == 0:
        w_0 = 116
        h_0 = 120
        cout_0 = 48
        #in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 256, num))).astype(int)+3,1)
        in_channels = np.expand_dims(np.linspace(3, 258, 256).astype(int), 1)
        out_channels = np.full((num, 1), cout_0)
        kernel_size = np.full((num, 1), 3)
        stride = np.full((num, 1), 1)
        #kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        #stride = np.expand_dims(np.random.randint(1,4,num),1)
        in_height = np.full((num, 1), h_0)
        in_width = np.full((num, 1), w_0)
        #out_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        #in_height = np.expand_dims(np.random.randint(8,513,num),1)
        #in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = np.concatenate((in_channels, out_channels, kernel_size, stride), axis=1)
        input_dim = np.concatenate((in_channels, in_width, in_height),axis=1)
        
    elif op_pairs[0] == 1:
        in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        kernel_size = np.full((num, 1), 2)
        stride = np.full((num, 1), 2)
        #kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        #stride = np.expand_dims(np.random.randint(1,4,num),1)
        config_1 = np.concatenate((kernel_size, stride), axis=1)
        input_dim = np.concatenate((in_channels, in_width, in_height),axis=1)

    elif op_pairs[0] == 2:
        in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        out_channels = in_channels
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = in_channels
        input_dim = np.concatenate((in_channels, in_width, in_height),axis=1)

    elif op_pairs[0] == 3:
        in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        out_channels = in_channels
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = np.zeros((num,1)).astype(int)
        input_dim = np.concatenate((in_channels, in_width, in_height),axis=1)

    elif op_pairs[0] == 4:
        in_channels = np.expand_dims(np.random.randint(3,32,num),1)
        in_height = np.expand_dims(np.random.randint(8,32,num),1)
        in_width = np.expand_dims(np.random.randint(8,32,num),1)
        config_1 = np.zeros((num,1)).astype(int)
        input_dim = np.concatenate((in_channels, in_width, in_height),axis=1)

    elif op_pairs[0] == 5:
        input_features = np.expand_dims(np.random.randint(3,513,num),1)
        output_features = np.expand_dims(np.random.randint(3,513,num),1)
        config_1 = np.concatenate((input_features, output_features), axis=1)
        input_dim = input_features

    return [config_1, input_dim]

def measurement_core(model, repetitions, input_dim, trt=False):
    """Measure model inference latency several times and calculate the numerical average.
    
    Args:
        model (object): Target model.
        repetitions (int): Measurement repetitions.
        input_dim (list): The dimension of input data.
    
    Returns:
        An average inference latency.
    """
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    dummy_input = torch.randn(input_dim, dtype=torch.float).to(device)
    #model_trt = torch2trt(model, [dummy_input])
    for rep in range(repetitions):
        torch.cuda.synchronize()
        starter.record()
        model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    return mean_syn

def ops_measurement(ops, set_num):
    """Measure inference latency set of different operators.
    
    Args:
        ops (list): A list of operators.
        set_num (int): Generated data set quantity.
    """
    repetitions = 200
    with open("./other_operators_tx2.csv", "wt", newline='') as csvfile:
        fieldnames = ["operator", "latency", "FLOPS"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        
        pbar0 = tqdm(total=len(ops), position=1, leave=False)
        op_num = 0
        with torch.no_grad():
            for op1 in range (len(ops)):
                pbar1 = tqdm(total=set_num, position=0, leave=False)
                op_pairs = [op1]
                if op1 != 1:
                    pass

                config_set = config_gen(op_pairs, set_num)
                for i in range (set_num):
                    model_op1 = model([op_pairs[0]], [config_set[0][i]]).to(device).eval()
                    dummy_input = torch.randn(np.insert((config_set[1][i]),0,1, axis=0).tolist(), dtype=torch.float).to(device)
                    #model_op1 = torch2trt(model_op1, [dummy_input])
                    latency_op1 = measurement_core(model_op1, repetitions, np.insert((config_set[1][i]),0,1, axis=0).tolist())
                    #flops, params = profile(model_op1, inputs=(dummy_input,))
                    #flops, params = get_model_complexity_info(model_op1, tuple(dummy_input.shape[1:]))
                    if op1 == 0:
                        flops = config_set[0][i][0]
                        pass
                    
                    elif op1 == 1:
                        in_channels = config_set[1][i][0]
                        in_height = config_set[1][i][1]
                        in_width = config_set[1][i][2]
                        flops = in_channels * in_height * in_width
                        pass
                    
                    elif op1 in [2,3]:
                        flops = config_set[1][i][0] * config_set[1][i][1] * config_set[1][i][2]
                    
                    pbar1.update(1)
                    pbar1.set_description("\rOperator: <{:s}>  seq: ({:d} / {:d})"
                      .format(ops[op1], i, set_num))
                    writer.writerow(
                        {
                            "operator": op_pairs[0],
                            "latency": latency_op1,
                            "FLOPS": flops
                        }
                    )
                    del model_op1
                    torch.cuda.empty_cache()
                pbar1.close()
                op_num += 1
                pbar0.update(1)
                pbar0.set_description("\rOverall progress: ({:d} / {:d})"
                  .format(op_num, len(ops)))

# Measured operator types.
ops = ['conv', 'pool', 'bn', 'relu']
# Generated data set quantity.
set_num = 256

warm_up(model)
ops_measurement(ops, set_num)