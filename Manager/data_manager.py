from threading import Thread
#from model_manager import Model_Manager
import torch
import numpy as np
import time

class Data_Manager():
    """Manage partial data distribution and collection.
    
    Attributes:
        local_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
        inter_commu (object): Internal communication module.
        concate_data (array): Data composed of several partial data.
    """
    def __init__(self, local_type, inter_commu, communication):
        self.local_type = local_type
        self.inter_commu = inter_commu
        self.ip = communication.ip
        self.concate_data = torch.empty(1, 1, 1, 1)
        thread = []
        thread.append(Thread(target = self.data_distribution))
        for t in thread:
            t.start()
        self.data_collection()

    def data_partition(self, input_data, range_lst):
        """Partition input data according to the range list.
        
        Args:
            input_data (array): Data to be partitioned.
            range_lst (list): A list of data range.

        Returns:
            A list of partitioned data.
        """
        data_lst = []
        for range in range_lst:
            partition_data = input_data[:,:,range[0]: range[1]+1, :]
            data_lst.append(partition_data)
        return data_lst
        
    def data_concate(self, data_lst, model_id):
        """Concatenate several partial data to a complete data.
        
        Args:
            data_lst (list): A list of partial data.
            model_id (int): A list of models' unique ids.
        
        Returns:
            A concatenated data.
        """
        if [self.concate_data.shape[1], self.concate_data.shape[2], self.concate_data.shape[3]] != self.inter_commu.out_range[self.inter_commu.model_id.index(model_id)]:
            self.concate_data = torch.empty(1, self.inter_commu.out_range[self.inter_commu.model_id.index(model_id)][0], self.inter_commu.out_range[self.inter_commu.model_id.index(model_id)][1], self.inter_commu.out_range[self.inter_commu.model_id.index(model_id)][2])
        s = 0
        for data in data_lst:
            self.concate_data[:,:,s:s+data.shape[2],:] = data
            s += data.shape[2]
        return self.concate_data
    
    def data_distribution(self):
        """Distribute input data to cluster members.

        The partition is based on the scheduling results.
        """
        if self.local_type[-1] == 'H':
            while True:
                while len(self.inter_commu.data_queue) == 0 or len(self.inter_commu.submodel_id) == 0 or len(self.inter_commu.candidate_queue)>15 or len(self.inter_commu.concate_partial)>15 or self.inter_commu.task_waiting.is_set():
                    time.sleep(0.001)
                if self.inter_commu.data_queue_lock.acquire():
                    data = self.inter_commu.data_queue.pop(0)
                    print("[3] Get a subdata")
                    self.inter_commu.data_queue_lock.release()
                data_id = data['data_id']
                model_id = data['model_id']
                input_data = data['content']
                current_inter_range = []
                for i_range in self.inter_commu.inter_range[self.inter_commu.model_id.index(model_id)]:
                    current_inter_range.append(i_range[0]) 
                data_lst = self.data_partition(input_data, current_inter_range)
                self.inter_commu.add_data_id(data_id)
                self.inter_commu.create_concate_queue(len(self.inter_commu.inter_range[self.inter_commu.model_id.index(model_id)]))
                for i, i_range in enumerate(self.inter_commu.inter_range[self.inter_commu.model_id.index(model_id)]):
                    data_s = {
                        'type': 'data',
                        'data_id': data_id,
                        'seq': i,
                        'submodel_id': self.inter_commu.model_submodel[self.inter_commu.model_id.index(model_id)][-1],
                        'content': data_lst[i]
                    }
                    if i == 0:
                        self.inter_commu.add_candidate([data_id, i, self.inter_commu.model_submodel[self.inter_commu.model_id.index(model_id)][-1], data_lst[i]])
                    else:
                        for addr in self.inter_commu.inter_addr[0]:
                            if addr[0] == i_range[1]:
                                self.inter_commu.send_data(data_s, addr[0])
                                print("Send a subtask.")
                                break
                        
                
        elif self.local_type[-1] == 'M':
            while True:
                while len(self.inter_commu.data_queue) == 0:
                    time.sleep(0.001)
                if self.inter_commu.data_queue_lock.acquire():
                    data = self.inter_commu.data_queue.pop(0)
                    print("Get a subdata.")
                    self.inter_commu.data_queue_lock.release()
                data_id = data['data_id']
                input_data = data['content']
                seq = data['seq']
                submodel_id = data['submodel_id']
                self.inter_commu.add_candidate([data_id, seq, submodel_id, input_data])

    def data_collection(self):
        """Collect partial results as a complete result."""
        if self.local_type[-1] == 'H':
            flag = True
            while True:
                while len(self.inter_commu.partial_queue) == 0:
                    time.sleep(0.001)
                if self.inter_commu.partial_queue_lock.acquire():
                    data = self.inter_commu.partial_queue.pop(0)
                    print("Get a raw data.")
                    self.inter_commu.partial_queue_lock.release()
                data_id = data['data_id']
                if not data_id in self.inter_commu.data_id:
                    continue
                seq = data['seq']
                partial_data = data['content']
                submodel_id = data['submodel_id']
                if self.inter_commu.concate_partial_lock.acquire():
                    temp_concate = self.inter_commu.concate_partial[self.inter_commu.data_id.index(data_id)]
                    temp_concate[seq] = partial_data
                    self.inter_commu.concate_partial[self.inter_commu.data_id.index(data_id)] = temp_concate
                    self.inter_commu.concate_partial_lock.release()
                if self.inter_commu.concate_partial[self.inter_commu.data_id.index(data_id)].count(None) == 0:
                    print("Data concated.")
                    if self.inter_commu.concate_partial_lock.acquire():
                        concated_data = self.inter_commu.concate_partial.pop(self.inter_commu.data_id.index(data_id))
                        self.inter_commu.concate_partial_lock.release()
                    if self.inter_commu.data_id_lock.acquire():
                        self.inter_commu.data_id.pop(self.inter_commu.data_id.index(data_id))
                        self.inter_commu.data_id_lock.release()
                    for i, submodel_id_lst in enumerate(self.inter_commu.model_submodel):
                        if submodel_id in submodel_id_lst:
                            addr = self.inter_commu.send_node[i]
                            model_id = self.inter_commu.model_id[i]
                            break
                    concated_data = self.data_concate(concated_data, model_id)
                    data = {
                        'type': 'result',
                        'data_id': data_id,
                        'model_id': model_id,
                        'content': concated_data
                    }
                    if data_id == 10:
                        tic = time.time()
                    elif data_id == 19:
                        toc = time.time()
                        self.inter_commu.local_cluster_ability.append(toc-tic)
                        for ch in self.inter_commu.ch_avi:
                            self.inter_commu.abi_cfm.append(False)
                            content = {
                                    'task_type': 'cluster ability',
                                    'ip': self.ip,
                                    'ability': toc-tic
                                }
                            data = {'type': 'task', 'content': content}
                            self.inter_commu.send_control_data(data, ch[0], ch[1])
                        time.sleep(5)
                        while self.inter_commu.abi_cfm.count(False) != 0:
                            for i, state in enumerate(self.inter_commu.abi_cfm):
                                if state == False:
                                    self.inter_commu.send_control_data(data, self.inter_commu.ch_avi[i][0], self.inter_commu.ch_avi[i][1])
                            time.sleep(5)
                    elif data_id == 29:
                        self.inter_commu.local_measure.set()
                    if addr == None:
                        if flag or data_id < 32:
                            flag = False
                        else:
                            latency = time.time() - tic2
                        tic2 = time.time()
                        continue
                    self.inter_commu.send_result(data, addr[0], addr[1])
                    print("Send a result.")

        if self.local_type[-1] == 'M':
            while True:
                while len(self.inter_commu.partial_queue) == 0:
                    time.sleep(0.001)
                if self.inter_commu.partial_queue_lock.acquire():
                    data = self.inter_commu.partial_queue.pop(0)
                    self.inter_commu.partial_queue_lock.release()
                submodel_id = data['submodel_id']
                addr = self.inter_commu.send_node[self.inter_commu.submodel_id.index(submodel_id)]
                self.inter_commu.send_data(data, addr[0])
                print("Send a result.")
