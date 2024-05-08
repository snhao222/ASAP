import os
import torch
from torch import nn
import numpy as np
import pickle
import yaml
from threading import Thread
import time

class PredictorModel(nn.Module):
    """Predictor model for latency fusion fine-tuning."""
    def __init__(self, device):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(12, 32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(32,1))
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

class Scheduler():
    """Allocate tasks among clusters and UAVs inside clusters.
    
    Attributes:
        inter_commu (object): Internal communication module.
        task_manager (object): Task manage module.
        model_manager (object): Model manage module.
        Task_generator: Task generate module.
        task_owner (bool): "True" when the local device is the task owner.
        communication (object): External communication module.
        state_manager (object): State manage module.
        Latency_predictor (object): Inference performance prediction module.
        parapath (str): Predictor configuration file path.
        fitpath (str): File path of basic latency pattern.
        task_type (str): Model name of task.
    """
    def __init__(self, inter_commu, task_manager, model_manager, task_owner, communication, state_manager, Latency_predictor,\
                 parapath, fitpath, Task_generator, task_type, elastic):

        self.inter_commu = inter_commu
        parapath = os.path.join(os.getcwd(), parapath)
        with open(parapath, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        fitpath = os.path.join(os.getcwd(), fitpath)
        with open(fitpath, 'r', encoding='utf-8') as f:
            fitting = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.predictor_lo = Latency_predictor(configs, fitting)

        self.cm_lst = []
        self.predictor = []
        while self.inter_commu.cm_config.count(None) != 0:
            time.sleep(0.001)
        for cm in self.inter_commu.cMem_list:
            self.cm_lst.append(cm['ip'])
            self.predictor.append(Latency_predictor(self.inter_commu.cm_config[self.cm_lst.index(cm['ip'])][0], self.inter_commu.cm_config[self.cm_lst.index(cm['ip'])][1]))
        self.ip = communication.ip
        self.port = communication.port
        self.task_owner = task_owner
        self.model_manager = model_manager
        self.task_manager = task_manager
        self.state_manager = state_manager
        self.elastic = elastic
        self.model_id = 1
        self.submodel_id = 0
        self.range_in = [3, 720, 1280]

        self.model_f = PredictorModel(torch.device('cuda')).eval()
        self.model_f.load_state_dict(torch.load("./model_data/fusion_predictor.pt"))
        self.scaler = pickle.load(open('./model_data/scaler_1.pkl', 'rb'))

        if task_owner:
            task_generator = Thread(target = Task_generator, name='Task Generator    ',\
                                         args=(task_type, inter_commu))
            task_generator.start()

        self.E2S()
    
    def E2S(self):
        """Elastic efficient scheduling.
        
        Control the overall scheduling strategy of swarm.
        """
        data_id = 0
        while self.inter_commu.cm_rate.count(None) != 0:
            time.sleep(0.1)
        print("Get all performance.")
        # Test rate
        for cm in self.inter_commu.cMem_list:
            x = torch.randn(1, 3, 360, 640, dtype=torch.float)
            data_s = {
                        'type': 'test',
                        'content': x
                    }
            self.inter_commu.send_data(data_s, cm['ip'])
            x = torch.randn(1, 3, 360, 640, dtype=torch.float)
            data_s = {
                        'type': 'test',
                        'content': x
                    }
            self.inter_commu.send_data(data_s, cm['ip'])

        temp_range = [self.range_in[0], self.range_in[1], self.range_in[2]]
        self.range_in = [3, 360, 640]
        
        if self.inter_commu.task_queue_lock.acquire():
            self.inter_commu.task_queue.append({
                'task_type': 'update model',
                'model_id': 0,
                'nm_lst': 'test',
                'range_in': self.range_in,
                'send_node': None
            })
            self.inter_commu.task_queue_lock.release()
        while not self.inter_commu.rescheduling.is_set():
            time.sleep(0.01)
        while self.inter_commu.cm_rate_0.count(None) != 0:
            time.sleep(0.01)
        
        self.IDLB()
        self.inter_commu.rescheduling.clear()
        print("IDLB over.")
        x = torch.randn(1, 3, 360, 640, dtype=torch.float)
        self.inter_commu.task_waiting.set()
        while self.inter_commu.task_waiting.is_set():
            time.sleep(0.1)
        for i in range(30):
            data = {
                    'data_id': data_id,
                    'model_id': 0,
                    'content': x
                }
            self.inter_commu.add_data(data)
            data_id += 1
        self.range_in = temp_range
        
        if self.task_owner:
            while (self.inter_commu.clusters_ability.count(None) != 0) or (len(self.inter_commu.local_cluster_ability)==0):
                time.sleep(0.1)
            print("Get all ability info.")
            
            self.EDLB()
            self.inter_commu.cluster_measure.set()

        if self.elastic:
            while True:
                #local_model
                if self.inter_commu.ch_unavi_event.is_set():
                    self.inter_commu.ch_unavi_event.clear()
                    self.EDLB()

                elif self.inter_commu.ch_recover_event.is_set():
                    self.inter_commu.ch_recover_event.clear()
                    self.EDLB()

                elif self.inter_commu.rescheduling.is_set():
                    self.inter_commu.task_waiting.set()
                    self.IDLB()
                    self.inter_commu.rescheduling.clear()
                else:
                    time.sleep(0.1)
        else:
            while True:
                #local_model
    
                if self.inter_commu.rescheduling.is_set():
                    self.inter_commu.task_waiting.set()
                    self.IDLB()
                    self.inter_commu.rescheduling.clear()
                else:
                    time.sleep(0.1)

    def IDLB(self):
        """Internal Cluster Load Balancing (ICLB)."""
        len_lst = []
        nm_lst = self.inter_commu.model[-1]
        range_in = self.inter_commu.range_in[-1]
        if nm_lst == 'test':
            nm_lst = self.model_manager.t_nm_lst
            type_lst = self.model_manager.t_type_lst
            para_lst = self.model_manager.t_para_lst
            range_lst, width_lst = self.model_manager.out_deduction_test(range_in, nm_lst)
        else:
            model = self.model_manager.model_gen(nm_lst)
            nm_lst, type_lst, para_lst, _ = self.model_manager.para_extractor(model)
            del model
            range_lst, width_lst = self.model_manager.out_deduction(range_in, nm_lst)
        out_range = range_lst[-1]
        self.inter_commu.out_range[len(self.inter_commu.model_id)-1] = out_range
        desired_range_lst = []
        #interface
        if len(self.inter_commu.cm_avi) == 0:
            desired_range_lst = [[0, out_range[1]-1, None]]
        else:
            self.inter_commu.inter_addr[0]=[]
            for cm in self.inter_commu.cm_avi:
                if self.inter_commu.inter_addr_lock.acquire():
                    temp = self.inter_commu.inter_addr[0]
                    temp.append(cm)
                    self.inter_commu.inter_addr[0] = temp
                    self.inter_commu.inter_addr_lock.release()

            range_in_0, _, range_in_lst_0 = self.model_manager.MRT([0, out_range[1]-1], [1, 0], nm_lst, type_lst, para_lst, width_lst)
            range_in_1, _, range_in_lst_1 = self.model_manager.MRT([0, int(np.floor(out_range[1]/2))], [1, 0], nm_lst, type_lst, para_lst, width_lst)
            range_in_2, _, range_in_lst_2 = self.model_manager.MRT([int(np.floor(out_range[1]/2))+1, out_range[1]-1], [1, 1], nm_lst, type_lst, para_lst, width_lst)

            # CH total latency
            t_0_0 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_0, nm_lst, type_lst, para_lst)
            t_0_1 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_1, nm_lst, type_lst, para_lst)
            t_0_2 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_2, nm_lst, type_lst, para_lst)
            t_0 = (t_0_1 + t_0_2 - t_0_0) * len(self.inter_commu.cm_avi) + t_0_0

            t_i = []
            # CM total latency
            for cm in self.inter_commu.cm_avi:
                latency_i_0 = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_0, nm_lst, type_lst, para_lst)
                latency_i_1 = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_1, nm_lst, type_lst, para_lst)
                latency_i_2 = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst_2, nm_lst, type_lst, para_lst)
                latency_i = (latency_i_1 + latency_i_2 - latency_i_0) * len(self.inter_commu.cm_avi) + latency_i_0
                t_i.append(latency_i)
            # Aligned latency
            t_align = 1/(np.sum(np.divide(1, t_i)) + 1/t_0)

            # CH data partition
            range_s = 0
            range_len = 0
            range_avi = out_range[1]
            error_0 = float("inf")
            error = float("inf")
            while error <= error_0:
                error_0 = error
                range_len += 1
                range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+range_len-1], [1, 0], nm_lst, type_lst, para_lst, width_lst)
                tau_0 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
                error = abs(tau_0 - t_align)

            len_lst.append(range_len-1)
            # CM data partition
            for cm in self.inter_commu.inter_addr[0]:
                range_s += (range_len - 1)
                range_avi -= range_len
                range_len = 0
                error_0 = float("inf")
                error = float("inf")
                if cm == self.inter_commu.inter_addr[0][-1]:
                    len_lst.append(out_range[1]-range_s)
                else:
                    while error <= error_0:
                        error_0 = error
                        range_len += 1
                        range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+range_len-1], [0, 0], nm_lst, type_lst, para_lst, width_lst)
                        tau_i = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
                        error = abs(tau_i - t_align)
                    len_lst.append(range_len-1)
            # Adjustment
            tau_lst = [0]*(len(self.inter_commu.inter_addr[0])+1)
            range_s = 0
            len_i = len_lst[0]
            range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+len_i-1], [1, 0], nm_lst, type_lst, para_lst, width_lst)
            tau_0 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
            tau_lst[0] = tau_0
            for i, cm in enumerate(self.inter_commu.inter_addr[0]):
                range_s += len_i
                len_i = len_lst[i+1]
                cm_rate = self.inter_commu.cm_rate_0[self.inter_commu.cm_ip.index(cm[0])]
                cm_time = self.inter_commu.cm_time[self.inter_commu.cm_ip.index(cm[0])]
                range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+len_i-1], [0, 0], nm_lst, type_lst, para_lst, width_lst)
                tau_i = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
                t_size = range_lst[0][0]*range_lst[0][2]*range_in_lst[0] + range_lst[-1][0]*range_lst[-1][2]*range_in_lst[-1]
                tau_s = t_size/(self.range_in[0]*self.range_in[1]*self.range_in[2])*cm_time*cm_rate/self.inter_commu.cm_rate[self.inter_commu.cm_ip.index(cm[0])]
                tau_lst[i+1] = tau_s+tau_i
            error = abs(max(tau_lst)- min(tau_lst))
            error_0 = float("inf")
            while error < error_0:
                error_0 = error
                max_seq = tau_lst.index(max(tau_lst))
                min_seq = tau_lst.index(min(tau_lst))
                len_lst[max_seq] -= 1
                len_lst[min_seq] += 1
                if len_lst[max_seq] <= 1:
                    break
                range_s = 0
                len_i = len_lst[0]
                range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+len_i-1], [1, 0], nm_lst, type_lst, para_lst, width_lst)
                tau_0 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
                tau_lst[0] = tau_0
                for i, cm in enumerate(self.inter_commu.inter_addr[0]):
                    range_s += len_i
                    len_i = len_lst[i+1]
                    cm_rate = self.inter_commu.cm_rate_0[self.inter_commu.cm_ip.index(cm[0])]
                    cm_time = self.inter_commu.cm_time[self.inter_commu.cm_ip.index(cm[0])]
                    range_in_, _, range_in_lst = self.model_manager.MRT([range_s, range_s+len_i-1], [0, 0], nm_lst, type_lst, para_lst, width_lst)
                    tau_i = self.predictor[self.cm_lst.index(cm[0])].block_predictor(self.model_f, self.scaler, range_lst, range_in_lst, nm_lst, type_lst, para_lst)
                    t_size = range_lst[0][0]*range_lst[0][2]*range_in_lst[0] + range_lst[-1][0]*range_lst[-1][2]*range_in_lst[-1]
                    tau_s = t_size/(self.range_in[0]*self.range_in[1]*self.range_in[2])*cm_time*cm_rate/self.inter_commu.cm_rate[self.inter_commu.cm_ip.index(cm[0])]
                    tau_lst[i+1] = tau_s+tau_i
                error = abs(max(tau_lst)- min(tau_lst))
            len_lst[max_seq] += 1
            len_lst[min_seq] -= 1
            range_s = 0
            len_i = len_lst[0]
            desired_range_lst.append([range_s, range_s+len_i-1, None])
            for i, cm in enumerate(self.inter_commu.inter_addr[0]):
                range_s += len_i
                len_i = len_lst[i+1]
                desired_range_lst.append([range_s, range_s+len_i-1, cm[0]])

        range_in_lst = []
        padding_lst = []
        for desired_range in desired_range_lst:
            p_slice = [0, 0]

            range_in, padding, _ = self.model_manager.MRT(desired_range, p_slice, nm_lst, type_lst, para_lst, width_lst)

            range_in_lst.append([range_in, desired_range[2]])
            padding_lst.append(padding)

        if self.inter_commu.task_queue_lock.acquire():
            self.inter_commu.task_queue.append({
                'task_type': 'update submodel',
                'submodel_id': self.submodel_id,
                'nm_lst': nm_lst,
                'padding': padding_lst[0],
                'send_node': None,
                'range_in': None
            })
            self.inter_commu.task_queue_lock.release()

        if self.inter_commu.inter_range_lock.acquire():
            self.inter_commu.inter_range[len(self.inter_commu.model_id)-1] = range_in_lst
            self.inter_commu.inter_range_lock.release()

        i = 1
        for cm in self.inter_commu.inter_addr[0]:
            self.task_manager.task_send('update submodel', cm[0], cm[1], self.inter_commu.model_id[-1], nm_lst, [self.ip, self.port], range_in = self.inter_commu.range_in[-1], padding = padding_lst[i])
            i += 1
        self.submodel_id += 1

    def EDLB(self):
        """External Cluster Load Balancing (ECLB)."""
        range_in = self.range_in
        ch_lst = []
        ability_lst =[]
        range_in_lst = []
        model_lst = []

        for ch in self.inter_commu.ch_avi:
            ch_lst.append(ch)
            i = 0
            for ch_ in self.inter_commu.cHead_list:
                if ch_['ip'] == ch[0]:
                    break
                i += 1
            ability_lst.append(self.inter_commu.clusters_ability[i])
        ability_lst.append(self.inter_commu.local_cluster_ability[-1])
        nm_lst = self.model_manager.nm_lst
        type_lst = self.model_manager.type_lst
        para_lst = self.model_manager.para_lst
        block_lst = self.model_manager.block_lst

        # baseline latency
        range_lst, _ = self.model_manager.out_deduction(range_in, nm_lst)
        t_0 = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, [x[1] for x in range_lst], nm_lst, type_lst, para_lst)
        # local cluster
        range_in_lst.append(range_in)
        t_lo = (np.divide(1, np.sum(np.divide(1, ability_lst)))) * t_0 / self.inter_commu.local_cluster_ability[-1]
        error_0 = float("inf")
        error = float("inf")
        m_seq_s = 0
        m_seq_len = 0
        block_seq = -1
        seq_avi = len(nm_lst)
        tau_0 = 0
        while error <= error_0:
            error_0 = error
            block_seq += 1
            m_seq_len += len(block_lst[block_seq])
            tau = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst, [x[1] for x in range_lst], nm_lst[m_seq_s: m_seq_s+m_seq_len], type_lst[m_seq_s: m_seq_s+m_seq_len], para_lst[m_seq_s: m_seq_s+m_seq_len])
            if tau < tau_0:
                continue
            tau_0 = tau
            error = abs(tau - t_lo)
        m_seq_len -= len(block_lst[block_seq])
        model_lst.append(nm_lst[m_seq_s: m_seq_s+m_seq_len])
        range_in, _ = self.model_manager.out_deduction(range_in, nm_lst[m_seq_s: m_seq_s+m_seq_len])
        range_in = range_in[-1]
        block_seq -= 1

        # other clusters
        for i, ch in enumerate(ch_lst):
            range_in_lst.append(range_in)
            t_i = (np.divide(1, np.sum(np.divide(1, ability_lst)))) * t_0 / ability_lst[i]
            m_seq_s += m_seq_len
            seq_avi -= m_seq_len
            m_seq_len = 0
            error_0 = float("inf")
            error = float("inf")
            if ch == ch_lst[-1]:
                model_lst.append(nm_lst[m_seq_s: ])
            else:
                tau_0 = 0
                while error <= error_0:
                    error_0 = error
                    block_seq += 1
                    m_seq_len += len(block_lst[block_seq])
                    tau_i = self.predictor_lo.block_predictor(self.model_f, self.scaler, range_lst[m_seq_s: m_seq_s+m_seq_len], [x[1] for x in range_lst[m_seq_s: m_seq_s+m_seq_len]], nm_lst[m_seq_s: m_seq_s+m_seq_len], type_lst[m_seq_s: m_seq_s+m_seq_len], para_lst[m_seq_s: m_seq_s+m_seq_len])
                    if tau_i < tau_0:
                        continue
                    tau_0 = tau_i
                    error = abs(tau_i - t_i)
                m_seq_len -= len(block_lst[block_seq])
                model_lst.append(nm_lst[m_seq_s: m_seq_s+m_seq_len])
                range_in, _ = self.model_manager.out_deduction(range_in, nm_lst[m_seq_s: m_seq_s+m_seq_len])
                range_in = range_in[-1]
                block_seq -= 1
        print("EDLB Over.")

        for i, ch in enumerate(ch_lst):
            i += 1
            addr = [ch[0], ch[1]]
            if ch == ch_lst[-1]:
                self.task_manager.task_send('update model', addr[0], addr[1], self.model_id, model_lst[i], None, range_in_lst[i])
            else:
                self.task_manager.task_send('update model', addr[0], addr[1], self.model_id, model_lst[i], ch_lst[i], range_in_lst[i])

        if self.inter_commu.task_queue_lock.acquire():
            self.inter_commu.task_queue.append({
                'task_type': 'update model',
                'model_id': self.model_id,
                'nm_lst': model_lst[0],
                'range_in': range_in_lst[0],
                'send_node': ch_lst[0]
            })
            self.inter_commu.task_queue_lock.release()
        
        self.model_id += 1
  