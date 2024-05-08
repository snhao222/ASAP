import pandas as pd
import numpy as np
import yaml

def rule_detection(ops, op1, op2, latency1, latency2, latency1_2):
    """Detect operator fusion rules.
    
    Compare latency of independent operators and continuous operator pair.

    Args:
        ops (list): A list of operator types.
        op1 (list): A list of the first operators.
        op2 (list): A list of the second operators.
        latency1 (list): A list of the first operators' inference latency.
        latency2 (list): A list of the second operators' inference latency.
        latency1_2 (list): A list of the continuous operator pairs' inference latency.
    """
    fuse_rule = []
    for op_1 in range(len(ops)):
        op1_seq = np.where(op1 == op_1)
        op2_par = op2[op1_seq]
        latency1_op1 = latency1[op1_seq]
        latency2_op1 = latency2[op1_seq]
        latency1_2_op1 = latency1_2[op1_seq]
        for op_2 in range(len(ops)):
            op2_seq = np.where(op2_par == op_2)
            if len(op2_seq[0]) == 0:
                continue
            latency_1 = latency1_op1[op2_seq]
            latency_2 = latency2_op1[op2_seq]
            latency_1_2 = latency1_2_op1[op2_seq]
            fuse_rate = np.average(np.divide((latency_1 + latency_2 - latency_1_2), np.minimum(latency_1, latency_2))) * 100
            print("op1:{}, op2:{}" .format(ops[op_1], ops[op_2]))
            print(fuse_rate)
            if fuse_rate > 50: # fusion detection threshold
                fuse_rule.append([ops[op_1], ops[op_2]])

    with open('fusion_rules.yml', 'w', encoding='utf-8') as f:
        yaml.dump(data={"fusion_rules":fuse_rule}, stream=f, allow_unicode=True)

# Data set path
path = "local_latency.csv"
df_data = pd.read_csv(path)
op1 = np.array(df_data['operator1'])
op2 = np.array(df_data['operator2'])
latency1 = np.array(df_data['latency1'])
latency2 = np.array(df_data['latency2'])
latency1_2 = np.array(df_data['latency1+2'])
# Operator types
ops = ['conv', 'pool', 'bn', 'relu', 'conv+pool']

rule_detection(ops, op1, op2, latency1, latency2, latency1_2)