import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import yaml

class ratio_measurement():
    """Fitting data with least squares.
    
    Attributes:
        Xi (list): A list of independent variables.
        Yi (list): A list of dependent variables.
    """
    def __init__(self, Xi, Yi):
        self.Xi = Xi
        self.Yi = Yi
        
    def func(self, p, x):
        k, b = p
        return k * x + b
    
    def error(self, p, x, y):
        return self.func(p, x) - y

    def calculate(self):
        Para = leastsq(self.error, [1, 1], args=(self.Xi, self.Yi))
        return Para

def ratio_fit(ops, op_num, latency, flops, ratio_measurement):
    """Fitting lines for operators' latency.
    
    Args:
        ops (list): A list of target operators.
        op_num (list): A list of operator types.
        latency (list): A list of operators' inference latency.
        flops (list): A list of operators' FLOPs.
        ratio_measurement (class): Class 'ratio_measurement'.
    """
    op_lst = []
    k_lst = []
    b_lst = []
    for op in range(len(ops)):
        op_seq = np.where(op_num == op)
        latency_0 = latency[op_seq]
        flops_0 = flops[op_seq]
        Para = ratio_measurement(flops_0, latency_0).calculate()
        print(Para)
        k, b = Para[0]
        print(ops[op]+"-"*20)
        print("k", k)
        print("b", b)
        plt.figure(figsize=(8, 6))  
        plt.scatter(flops_0, latency_0, color="green", label="samples", linewidth=2)

        x = np.array(np.linspace(0, max(flops_0), 300))  
        y = k * x + b 
        plt.plot(x, y, color="red", label="fitting line", linewidth=2)
        plt.title('y={}+{}x'.format(b,k))
        plt.legend(loc='lower right')  
        plt.show()

        op_lst.append(ops[op])
        k_lst.append(float(k))
        b_lst.append(float(b))
    ydata = {
        'op_lst': op_lst,
        'k_lst': k_lst,
        'b_lst': b_lst
    }
    with open('fitting(tx2).yml', 'w', encoding='utf-8') as f:
        yaml.dump(data=ydata, stream=f, allow_unicode=True)

ops = ['conv', 'pool', 'bn', 'relu']
path = "other_operators_tx2.csv"
df_data = pd.read_csv(path)
op_num = np.array(df_data['operator'])
latency = np.array(df_data['latency'])
flops = np.array(df_data['FLOPS'])

plt.rcParams['font.sans-serif']=['SimHei']

ratio_fit(ops, op_num, latency, flops, ratio_measurement)