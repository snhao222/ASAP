import torch
import torch.nn as nn
import numpy as np
import csv
from tqdm import tqdm, trange
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
            elif op == 4:
                self.model.add_module('flatten',nn.Flatten())
            elif op ==5:
                self.model.add_module('{0}_fc'.format(i), nn.Linear(configrations[i][0], configrations[i][1]))
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

def config_gen(op_pairs, num):
    """Generate random model configurations for operator pairs.

    Args:
        op_pairs (list): A list of operator types.
        num (int): Generated data set quantity.
    """
    if op_pairs[0] == 0:
        #in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        #out_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        in_channels = np.expand_dims(np.random.randint(3,150,num),1)
        out_channels = np.expand_dims(np.random.randint(3,150,num),1)
        kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        stride = np.expand_dims(np.random.randint(1,3,num),1)
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = np.concatenate((in_channels, out_channels, kernel_size, stride), axis=1)
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        out_height = np.floor((in_height+2-kernel_size)/stride+1).astype(int)
        out_weight = np.floor((in_width+2-kernel_size)/stride+1).astype(int)
        output_dim = np.concatenate((out_channels, out_height, out_weight),axis=1)
        
    elif op_pairs[0] == 1:
        #in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        in_channels = np.expand_dims(np.random.randint(3,150,num),1)
        out_channels = in_channels
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        stride = np.expand_dims(np.random.randint(1,3,num),1)
        config_1 = np.concatenate((kernel_size, stride), axis=1)
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        out_height = np.floor((in_height-kernel_size)/stride+1).astype(int)
        out_weight = np.floor((in_width-kernel_size)/stride+1).astype(int)
        output_dim = np.concatenate((in_channels, out_height, out_weight),axis=1)
    elif op_pairs[0] == 2:
        #in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        in_channels = np.expand_dims(np.random.randint(3,150,num),1)
        out_channels = in_channels
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = in_channels
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        output_dim = input_dim
    elif op_pairs[0] == 3:
        #in_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        in_channels = np.expand_dims(np.random.randint(3,150,num),1)
        out_channels = in_channels
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = np.zeros((num,1)).astype(int)
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        output_dim = input_dim
    elif op_pairs[0] == 4:
        in_channels = np.expand_dims(np.random.randint(3,32,num),1)
        in_height = np.expand_dims(np.random.randint(8,32,num),1)
        in_width = np.expand_dims(np.random.randint(8,32,num),1)
        config_1 = np.zeros((num,1)).astype(int)
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        output_dim = in_channels*in_height*in_width.astype(int)
    elif op_pairs[0] == 5:
        input_features = np.expand_dims(np.random.randint(3,513,num),1)
        output_features = np.expand_dims(np.random.randint(1,513,num),1)
        config_1 = np.concatenate((input_features, output_features), axis=1)
        input_dim = input_features
        output_dim = output_features
    elif len(op_pairs[0]) == 2:
        in_channels = np.expand_dims(np.random.randint(3,150,num),1)
        out_channels = np.expand_dims(np.random.randint(3,150,num),1)
        kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        stride = np.expand_dims(np.random.randint(1,3,num),1)
        in_height = np.expand_dims(np.random.randint(8,513,num),1)
        in_width = np.expand_dims(np.random.randint(8,513,num),1)
        config_1 = np.concatenate((in_channels, out_channels, kernel_size, stride), axis=1)
        input_dim = np.concatenate((in_channels, in_height, in_width),axis=1)
        in_height_2 = np.floor((in_height+2-kernel_size)/stride+1).astype(int)
        in_width_2 = np.floor((in_width+2-kernel_size)/stride+1).astype(int)
        #output_dim = np.concatenate((in_channels, out_height, out_weight),axis=1)
        kernel_size_2 = np.expand_dims(np.random.randint(1,4,num),1)
        stride_2 = np.expand_dims(np.random.randint(1,3,num),1)
        config_2 = np.concatenate((kernel_size_2, stride_2), axis=1)
        out_height_2 = np.floor((in_height_2-kernel_size_2)/stride_2+1).astype(int)
        out_width_2 = np.floor((in_width_2-kernel_size_2)/stride_2+1).astype(int)
        output_dim = np.concatenate((out_channels, out_height_2, out_width_2),axis=1)
        config_1 = [config_1, config_2]

    
    if op_pairs[1] == 0:
        in_channels = out_channels
        #out_channels = np.expand_dims(np.floor(abs(np.random.normal(0, 170, num))).astype(int)+3,1)
        out_channels = np.expand_dims(np.random.randint(3,150,num),1)
        kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        stride = np.expand_dims(np.random.randint(1,3,num),1)
        config_2 = np.concatenate((in_channels, out_channels, kernel_size, stride), axis=1)
    elif op_pairs[1] == 1:
        kernel_size = np.expand_dims(np.random.randint(1,4,num),1)
        stride = np.expand_dims(np.random.randint(1,3,num),1)
        config_2 = np.concatenate((kernel_size, stride), axis=1)
    elif op_pairs[1] == 2:
        config_2 = out_channels
    elif op_pairs[1] == 3:
        config_2 = np.zeros((num,1)).astype(int)
    elif op_pairs[1] == 4:
        config_2 = np.zeros((num,1)).astype(int)
    elif op_pairs[1] == 5:
        input_features = output_dim
        output_features = np.expand_dims(np.random.randint(1,513,num),1)
        config_2 = np.concatenate((input_features, output_features), axis=1)

    return [config_1, config_2, input_dim, output_dim]

def measurement_core(model, repetitions, input_dim, trt=False):
    """Measure the model inference latency several times and calculate the numerical average.
    
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

def ops_measurement(ops, test_set, set_num):
    """Measure inference latency set of operator pairs.
    
    Args:
        ops (list): A list of operator types.
        test_set (list): A list of operator pairs to be measured.
        set_num (int): Generated data set quantity.
    """
    repetitions = 500

    with open("./local_latency_test.csv", "wt", newline='') as csvfile:
        fieldnames = ["operator1", "operator2", "latency1", "latency2", "latency1+2"]
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        
        pbar0 = tqdm(total=len(ops)*len(ops), position=1, leave=False)
        op_num = 0

        with torch.no_grad():
            for op1, op2 in test_set:
                #print('op1 {}, op2 {} len{}' .format(op1, op2, len(op1)))
                pbar1 = tqdm(total=set_num, position=0, leave=False)
                if not isinstance(op1, list):
                    op_pairs = [ops.index(op1), ops.index(op2)]
                    config_set = config_gen(op_pairs, set_num)
                    for i in range (set_num):
                        model_op1 = model([op_pairs[0]], [config_set[0][i]]).to(device).eval()
                        model_op2 = model([op_pairs[1]], [config_set[1][i]]).to(device).eval()
                        model_op12 = model(op_pairs, [config_set[0][i], config_set[1][i]]).to(device).eval()
                        latency_op1 = measurement_core(model_op1, repetitions, np.insert((config_set[2][i]),0,1, axis=0).tolist())
                        latency_op2 = measurement_core(model_op2, repetitions, np.insert((config_set[3][i]),0,1, axis=0).tolist())
                        latency_op12 = measurement_core(model_op12, repetitions, np.insert((config_set[2][i]),0,1, axis=0).tolist())
                        pbar1.update(1)
                        pbar1.set_description("\rOperators: <{:s}, {:s}>  seq: ({:d} / {:d})"
                          .format(op1, op2, i, set_num))
                        writer.writerow(
                            {
                                "operator1": op_pairs[0],
                                "operator2": op_pairs[1],
                                "latency1": latency_op1,
                                "latency2": latency_op2,
                                "latency1+2": latency_op12
                            }
                        )
                    torch.cuda.empty_cache()

                else:
                    op_pairs = [[ops.index(op1[0]), ops.index(op1[1])], ops.index(op2)]
                    config_set = config_gen(op_pairs, set_num)

                    for i in range (set_num):
                        model_op1 = model(op_pairs[0], [config_set[0][0][i], config_set[0][1][i]]).to(device).eval()
                        model_op2 = model([op_pairs[1]], [config_set[1][i]]).to(device).eval()
                        model_op12 = model([op_pairs[0][0], op_pairs[0][1], op_pairs[1]], [config_set[0][0][i], config_set[0][1][i], config_set[1][i]]).to(device).eval()

                        latency_op1 = measurement_core(model_op1, repetitions, np.insert((config_set[2][i]),0,1, axis=0).tolist())
                        latency_op2 = measurement_core(model_op2, repetitions, np.insert((config_set[3][i]),0,1, axis=0).tolist())
                        latency_op12 = measurement_core(model_op12, repetitions, np.insert((config_set[2][i]),0,1, axis=0).tolist())

                        pbar1.update(1)
                        pbar1.set_description("\rOperators: <{:s}, {:s}>  seq: ({:d} / {:d})"
                          .format(op1[0]+op1[1], op2, i, set_num))
                        writer.writerow(
                            {
                                "operator1": len(ops),
                                "operator2": op_pairs[1],
                                "latency1": latency_op1,
                                "latency2": latency_op2,
                                "latency1+2": latency_op12
                            }
                        )
                    torch.cuda.empty_cache()

                pbar1.close()
                op_num += 1
                pbar0.update(1)
                pbar0.set_description("\rOverall progress: ({:d} / {:d})"
                  .format(op_num, 14))

# Operator types
ops = ['conv', 'pool', 'bn', 'relu']
# Operator types
test_set = [['conv', 'conv'], ['conv', 'pool'], ['conv', 'bn'], ['conv', 'relu'], ['pool', 'conv'], ['pool', 'bn'], ['pool', 'relu'],\
             ['bn', 'conv'], ['bn', 'pool'], ['bn', 'relu'], ['relu', 'conv'], ['relu', 'pool'], ['relu', 'bn'], [['conv', 'relu'], 'pool']]

set_num = 10
print('\r\n')
ops_measurement(ops, test_set, set_num)
