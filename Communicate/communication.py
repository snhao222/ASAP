import sys
import socket
import torch
import struct
import time
import multiprocessing
import pickle
import gzip

buffersend = 60002
buffersize = 60006

class Communication(object):
    """External communication module.

    All the messages exchange between devices depends on this module.

    Attributes:
        ip (str): Local ip.
        port (int): Local command transmission port.
        server_socket: UDP socket of server.
        client_socket: UDP socket of client.
        inter_commu (object): Internal communication module.
    """
    def __init__(self, inter_commu, port = ''):
        ip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip_socket.connect(('8.8.8.8',80))
        self.ip = ip_socket.getsockname()[0]
        self.port = port
        self.key = 2221
        self.confirm = 2333
        self.c_socket = ''
        self.server_socket = self.create_server_socket()
        self.client_socket = self.create_client_socket()
        self.inter_commu = inter_commu
        
    def create_server_socket(self):
        """Create a server socket.
        
        The socket is bound to the command transmission port of local ip.
        """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)      
        server_socket.bind((self.ip, self.port))                                             
        return server_socket
    
    def create_client_socket(self):
        """Create a client socket."""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)          
        return client_socket

    def get_message(self):
        """Receive messages from other devices.
        
        Returns:
            ultimate_buffer (bytes): A byte list of received raw message.
        """
        ultimate_buffer = b''
        s = 1
        err = 0
        thr = 30
        while True:
            data, a  = self.server_socket.recvfrom(65506)
            if len(data) != 8:
                continue
            seq = data[0:4]
            if struct.unpack('i',seq)[0] == s:
                data_len = data[4:]
                data_len = struct.unpack('i',data_len)[0]
                self.server_socket.sendto(struct.pack('i',s),a)
                s += 1
                break
            else:
                continue
        if data_len:
            while data_len >0 :
                if err > thr:
                    break
                if data_len >= buffersize:
                    content, a = self.server_socket.recvfrom(65506)
                    seq = content[0:4]
                    if struct.unpack('i',seq)[0] == s:
                        self.server_socket.sendto(seq,a)
                        ultimate_buffer += content[4:]
                        data_len -= buffersend
                        s += 1
                    elif struct.unpack('i',seq)[0] < s:
                        self.server_socket.sendto(seq, a)
                    else:
                        err += 1
                        continue
                else:
                    content, a = self.server_socket.recvfrom(65506)
                    seq = content[0:4]
                    if struct.unpack('i',seq)[0] == s:
                        self.server_socket.sendto(seq,a)
                        ultimate_buffer += content[4:]
                        #self.server_socket.sendto(seq,a)
                        #self.server_socket.sendto(seq,a)
                        break
                    elif struct.unpack('i',seq)[0] < s:
                        self.server_socket.sendto(seq, a)
                    else:
                        err += 1
                        continue
        return ultimate_buffer
    
    def send_message(self, message_to_send, ip='', port=''):
        """Send message to other devices.

        Args:
            message_to_send (bytes): A byte list of raw message which is sent to other devices.
            ip (str): Target ip.
            port (int):Target port.
        """
        i = 0
        s = 1
        err = 0
        thr = 30
        data_len = len(message_to_send)
        pack_len = struct.pack('i',data_len)
        pack_len = struct.pack('i',s)+pack_len
        self.client_socket.settimeout(0.3)
        
        try:
            self.client_socket.sendto(pack_len, (ip , port))
        except:
            self.client_socket.sendto(pack_len, (ip , port))
        while True:
            if err > thr:
                break
            try:
                ack , _ = self.client_socket.recvfrom(65506)
            except:
                while True:
                    try:
                        self.client_socket.sendto(pack_len, (ip , port))
                        break
                    except:
                        continue
            else:
                if struct.unpack('i',ack)[0] == s:
                    s += 1
                    tic = time.time()
                    break
                else:
                    continue
        while data_len > 0:
            if err > thr:
                break
            if data_len >= buffersend:
                content = message_to_send[i*buffersend : (i+1)*buffersend]
                content = struct.pack('i',s)+content
                while True:
                    try:
                        self.client_socket.sendto(content, (ip , port))
                    except:
                        try:
                            self.client_socket.sendto(content, (ip , port))
                        except:
                            break
                    try:
                        ack, _ = self.client_socket.recvfrom(65506)
                    except:
                        err += 1
                        if err > thr:
                            break
                        continue
                    else:
                        if struct.unpack('i',ack)[0] == s:
                            data_len -= buffersend
                            i += 1
                            s += 1
                            break
                        else:
                            continue
                
            else:
                content = message_to_send[i*buffersend :]
                content = struct.pack('i',s)+content
                while True:
                    try:
                        self.client_socket.sendto(content, (ip , port))
                    except:
                        try:
                            self.client_socket.sendto(content, (ip , port))
                        except:
                            break
                    try:
                        ack, _ = self.client_socket.recvfrom(65506)
                    except:
                        err += 1
                        if err > thr:
                            break
                        continue
                    else:
                        if struct.unpack('i',ack)[0] == s:
                            break
                        else:
                            continue
                break
        toc =time.time()
        if ip in self.inter_commu.ch_ip:
            if self.inter_commu.rate[self.inter_commu.ch_ip.index(ip)] != 0:
                self.inter_commu.rate[self.inter_commu.ch_ip.index(ip)] = int(self.inter_commu.rate[self.inter_commu.ch_ip.index(ip)]*0.8 + (len(message_to_send)/ (toc-tic))*0.2)
            else:
                self.inter_commu.rate[self.inter_commu.ch_ip.index(ip)] = int(len(message_to_send)/ (toc-tic))
        elif ip in self.inter_commu.cm_ip:
            if self.inter_commu.cm_time[self.inter_commu.cm_ip.index(ip)] == 0:
                self.inter_commu.cm_time[self.inter_commu.cm_ip.index(ip)] = (toc-tic)*1000
                self.inter_commu.cm_rate_0[self.inter_commu.cm_ip.index(ip)] = int(len(message_to_send)/ (toc-tic))
        print("Send Time:"+str((toc-tic)*1000)+"ms")
    
    def confirm_connection(self):
        while True:
            data, a  = self.client_socket.recvfrom(4)
            self.c_socket = a
            if struct.unpack('i',data)[0] == self.key:
                self.client_socket.sendto(struct.pack('i',self.confirm),a)
                self.client_socket.sendto(struct.pack('i',self.confirm),a)
                self.client_socket.sendto(struct.pack('i',self.confirm),a)
                break
            else:
                continue

    def connection_establish(self, ip, port):
        load = struct.pack('i',self.key)
        self.client_socket.settimeout(0.3)
        self.client_socket.sendto(load, (ip , port))
        while True:
            try:
                ack , _ = self.client_socket.recvfrom(4)
            except:
                self.client_socket.sendto(load, (ip , port))
            else:
                if struct.unpack('i',ack)[0] == self.confirm:
                    print("—— Connection Established ——")
                    break
                else:
                    continue

class Inter_commu():
    """Internal communication module.

    All the messages exchange between modules depends on this module.
    
    Attributes:
        receive_control_queue (list): A list of received control messages.
        receive_queue (list): A list of received data.
        receive_result_queue (list): A list of received partial results from cluster members.
        send_queue (list): A list of data to be sent to other devices.
        send_control_queue (list): A list of control messages to be sent to other devices.
        task_queue (list): A list of received tasks.
        data_queue (list): A list of received results or partial data.
        candidate_queue (list): A list of candidate data to be computed, including data id, sequence, and submodel id.
        results_queue (list): A list of computing results.
        partial_queue (list): A list of computing results under different partial data.
        cHead_list (list): A list of cluster heads.
        cMem_list (list): A list of cluster members.
        rate (list): Communication rate between cluster heads.
        cm_rate (list): Communication rate between cluster members.
        cm_rate_0 (list): Communication rate base.
        ch_avi (list): A list of current available cluster heads.
        cm_avi (list): A list of current available cluster members.
        clusters_ability (list): A list of other clusters' computing ability.
        local_cluster_ability (list): The computing ability of local cluster.
        submodel (list): A list of submodels' parameters.
        model_id (list): A list of models' unique ids.
        model (list): A list of the complete models.
        submodel_id (list): A list of submodels' unique ids.
        model_submodel (list): A list used to match models and submodels.
        out_range (list): A list discribes the output range of cluster members.
        data_id (list): A list of data's unique id.
        send_node (list): The target node for data transmission.
        concate_partial (list): Concatenate partial results.
        range_in (list): The input range of the complete model.
        inter_range (list): A list contains the input range for cluster members.
        inter_addr (list): A list contains collaborative cluster members.
    """
    def __init__(self):
        ##### List #####
        self.perfo_queue            =   multiprocessing.Manager().list()
        self.perfo_queue_lock       =   multiprocessing.Lock()

        self.receive_control_queue  =   multiprocessing.Manager().list()
        self.receive_control_lock   =   multiprocessing.Lock()

        self.receive_queue          =   multiprocessing.Manager().list()
        self.receive_queue_lock     =   multiprocessing.Lock()

        self.receive_result_queue  =   multiprocessing.Manager().list()
        self.receive_result_lock   =   multiprocessing.Lock()

        self.send_queue             =   multiprocessing.Manager().list()
        self.send_queue_lock        =   multiprocessing.Lock()

        self.send_control_queue     =   multiprocessing.Manager().list()
        self.send_control_queue_lock=   multiprocessing.Lock()

        self.task_queue             =   multiprocessing.Manager().list()
        self.task_queue_lock        =   multiprocessing.Lock()

        self.data_queue             =   multiprocessing.Manager().list()
        self.data_queue_lock        =   multiprocessing.Lock()

        self.alloc_queue            =   multiprocessing.Manager().list()
        self.alloc_queue_lock       =   multiprocessing.Lock()

        self.candidate_queue        =   multiprocessing.Manager().list()
        self.candidate_queue_lock   =   multiprocessing.Lock()

        self.results_queue          =   multiprocessing.Manager().list()
        self.results_queue_lock     =   multiprocessing.Lock()

        self.partial_queue          =   multiprocessing.Manager().list()
        self.partial_queue_lock     =   multiprocessing.Lock()

        ##### Event #####
        self.terminate              =   multiprocessing.Event()
        self.start_event            =   multiprocessing.Event()
        self.recv_confirm_event     =   multiprocessing.Event()
        self.perf_vary_event        =   multiprocessing.Event()
        self.ch_unavi_event         =   multiprocessing.Event()
        self.ch_recover_event       =   multiprocessing.Event()
        self.cm_unavi_event         =   multiprocessing.Event()
        self.cm_recover_event       =   multiprocessing.Event()
        self.rescheduling           =   multiprocessing.Event()
        self.task_waiting           =   multiprocessing.Event()
        self.warm_up                =   multiprocessing.Event()
        self.cluster_measure        =   multiprocessing.Event()
        self.local_measure          =   multiprocessing.Event()
        self.config_cfm             =   multiprocessing.Event()

        ##### Nodes Info. #####
        self.cHead_list             =   multiprocessing.Manager().list()
        self.cHead_lock             =   multiprocessing.Lock()

        self.cMem_list              =   multiprocessing.Manager().list()
        self.cMem_lock              =   multiprocessing.Lock()

        self.ground_station         =   multiprocessing.Manager().list()
        self.ground_station_lock    =   multiprocessing.Lock()

        self.rate                   =   multiprocessing.Manager().list()
        self.cm_rate                =   multiprocessing.Manager().list()
        self.cm_rate_0              =   multiprocessing.Manager().list()
        self.cm_time                =   multiprocessing.Manager().list()

        self.ch_perfo               =   multiprocessing.Manager().list()
        self.ch_ip                  =   multiprocessing.Manager().list()

        self.cm_perfo               =   multiprocessing.Manager().list()
        self.cm_ip                  =   multiprocessing.Manager().list()

        self.ch_avi                 =   multiprocessing.Manager().list()
        self.ch_avi_lock            =   multiprocessing.Lock()

        self.abi_cfm                =   multiprocessing.Manager().list()

        self.cm_avi                 =   multiprocessing.Manager().list()
        self.cm_avi_lock            =   multiprocessing.Lock()

        self.clusters_ability       =   multiprocessing.Manager().list()
        
        self.local_cluster_ability  =   multiprocessing.Manager().list()

        self.cm_config              =   multiprocessing.Manager().list()
        self.cm_config_lock         =   multiprocessing.Lock()

        self.data_port              =   multiprocessing.Manager().list()

        ##### Model #####
        self.complete_model         =   multiprocessing.Manager().list()
        self.complete_model_lock    =   multiprocessing.Lock()

        self.submodel               =   multiprocessing.Manager().list()
        self.submodel_lock          =   multiprocessing.Lock()

        self.model_id               =   multiprocessing.Manager().list()
        self.model_id_lock          =   multiprocessing.Lock()

        self.model                  =   multiprocessing.Manager().list()
        self.model_lock             =   multiprocessing.Lock()

        self.submodel_id            =   multiprocessing.Manager().list()
        self.submodel_id_lock       =   multiprocessing.Lock()

        self.model_submodel         =   multiprocessing.Manager().list()
        self.model_submodel_lock    =   multiprocessing.Lock()

        self.out_range              =   multiprocessing.Manager().list()
        self.out_range_lock         =   multiprocessing.Lock()

        ##### Data & Sequence #####
        self.data_id                =   multiprocessing.Manager().list()
        self.data_id_lock           =   multiprocessing.Lock()

        self.send_node              =   multiprocessing.Manager().list()
        self.send_node_lock         =   multiprocessing.Lock()

        self.concate_partial        =   multiprocessing.Manager().list()
        self.concate_partial_lock   =   multiprocessing.Lock()

        self.range_in               =   multiprocessing.Manager().list()
        self.range_in_lock          =   multiprocessing.Lock()
        self.range_in.append([])

        ##### Scheduling Info. #####
        self.inter_range            =   multiprocessing.Manager().list()
        self.inter_range_lock       =   multiprocessing.Lock()

        self.inter_addr             =   multiprocessing.Manager().list()
        self.inter_addr_lock        =   multiprocessing.Lock()
        self.inter_addr.append([])
    
    def updata_perfo(self, frq, mem_avi, task_num):
        if self.perfo_queue_lock.acquire():
            self.perfo_queue[0]={
                'frq': frq,
                'mem_avi': mem_avi,
                'task_num': task_num
                }
            self.perfo_queue_lock.release()
        self.perf_vary_event.set()

    def send_control_data(self, data, ip, port):
        """Add a new control data to candidate queue.
        
        Args:
            data (dict): Control data to be sent.
            ip (str): target ip address.
            port (int): target control messages transmission port.
        """
        if self.send_control_queue_lock.acquire():
            self.send_control_queue.append([pickle.dumps(data), ip, port])
            self.send_control_queue_lock.release()
    
    def send_data(self, data, ip):
        """Send a new data.
        
        Args:
            data (dict): Input data for cluster members.
            ip (str): target ip address.
        """
        if self.send_queue_lock.acquire():
            if ip in self.ch_ip:
                rate = self.rate[self.ch_ip.index(ip)]
            elif ip in self.cm_ip:
                rate = self.cm_rate[self.cm_ip.index(ip)]
            t_time = len(pickle.dumps(data))/rate
            if t_time > 0.5:
                scale_size = 0.4
            elif t_time > 0.3:
                scale_size = 0.3
            elif t_time > 0.1:
                scale_size = 0.2
            else:
                scale_size = 0.1
            #print("[Data]-- ip:{}, time:{}, scale:{}" .format(ip, t_time, scale_size))
            quantization = torch.quantize_per_tensor(data['content'], scale = scale_size, zero_point = 8, dtype=torch.quint8)
            data['content'] = quantization
            self.send_queue.append([gzip.compress(pickle.dumps(data), 2), ip])
            self.send_queue_lock.release()
    
    def send_result(self, data, ip, port):
        """Send a new result data.
        
        Args:
            data (dict): Result data to be sent.
            ip (str): target ip address.
            port (int): target control messages transmission port. The result messages transmission port equals to the comtrol port plus one.
        """
        if ip in self.ch_ip:
            rate = self.rate[self.ch_ip.index(ip)]
        elif ip in self.cm_ip:
            rate = self.cm_rate[self.cm_ip.index(ip)]
        t_time = len(pickle.dumps(data))/rate
        if t_time > 0.5:
            scale_size = 0.4
        elif t_time > 0.3:
            scale_size = 0.3
        elif t_time > 0.1:
            scale_size = 0.2
        else:
            scale_size = 0.1
        quantization = torch.quantize_per_tensor(data['content'], scale = scale_size, zero_point = 8, dtype=torch.quint8)
        data['content'] = quantization
        if self.results_queue_lock.acquire():
            self.results_queue.append([gzip.compress(pickle.dumps(data), 2), ip, port+1])
            print("[Send Result Queue:]", len(self.results_queue))
            self.results_queue_lock.release()

    def add_data(self, data):
        """Add a new data information to candidate queue.
        
        Including the input data, data id, data sequence, and submodel id.

        Args:
            data (array): Data information.
        """
        if self.data_queue_lock.acquire():
            self.data_queue.append(data)
            self.data_queue_lock.release()

    def add_candidate(self, data):
        """Add a new data to be computed.

        Args:
            data (tuple): Data to be computed.
        """
        if self.candidate_queue_lock.acquire():
            self.candidate_queue.append(data)
            self.candidate_queue_lock.release()

    def add_data_id(self, data_id):
        if self.data_id_lock.acquire():
            self.data_id.append(data_id)
            self.data_id_lock.release()
    
    def add_model_id(self, model_id):
        if self.model_id_lock.acquire():
            self.model_id.append(model_id)
            self.model_id_lock.release()

    def add_model(self, nm_lst):
        if self.model_lock.acquire():
            self.model.append(nm_lst)
            self.model_lock.release()
    
    def add_submodel(self, submodel):
        if self.submodel_lock.acquire():
            self.submodel.append(submodel)
            self.submodel_lock.release()
    
    def add_model_submodel(self, submodel_id):
        """Add model ids corresponding to submodels.

        Args:
            submodel_id (int): Id of submodel.
        """
        if self.model_submodel_lock.acquire():
            if submodel_id == []:
                self.model_submodel.append([])
            else:
                temp = self.model_submodel[-1]
                temp.append(submodel_id)
                self.model_submodel[-1] = temp
            self.model_submodel_lock.release()
    
    def add_submodel_id(self, submodel):
        if self.submodel_id_lock.acquire():
            self.submodel_id.append(submodel)
            self.submodel_id_lock.release()
    
    def add_send_node(self, send_node):
        if self.send_node_lock.acquire():
            self.send_node.append(send_node)
            self.send_node_lock.release()

    def update_rangein(self, range_in):
        """Update the input range.

        Args:
            range_in (list): Input range.
        """
        if self.range_in_lock.acquire():
            self.range_in[0] = range_in
            self.range_in_lock.release()

    def create_concate_queue(self, l):
        """Create null lists to place partial results from cluster members.
        
        Args:
            l (int): Number of nodes participating in the collaborative computing.
        """
        if self.concate_partial_lock.acquire():
            self.concate_partial.append([None]*l)
            self.concate_partial_lock.release()

    def get_candidate(self):
        """Extract a candidate computing data from candidate queue.
        
        The extracted data is ready to be computed.

        Returns:
            A set of data from the candidate queue.
        """
        if self.candidate_queue_lock.acquire():
            candidate = self.candidate_queue.pop(0)
            self.candidate_queue_lock.release()
        return candidate

    def store_model(self, model):
        if self.infer_model_lock.acquire():
            if len(self.infer_model) ==0:
                self.infer_model.append(model)
            else:
                self.infer_model[0] = model
            self.infer_model_lock.release()

    def get_task(self):
        """Extract a task from task queue.
        
        Returns:
            A task from task queue.
        """
        if self.task_queue_lock.acquire():
            task = self.task_queue.pop(0)
            self.task_queue_lock.release()
        return task

    def confirm_all(self):
        """Notify all the nodes in the swarm that all the collaborative members are already in place."""
        self.start_event.set()
        for ch in self.cHead_list:
            self.send_control_data({"type":"confirm connection", "content":"all in place"}, ch['ip'], ch['port'])
        for cm in self.cMem_list:
            self.send_control_data({"type":"confirm connection", "content":"all in place"}, cm['ip'], cm['port'])

    def confirm_members(self):
        """Notify cluster members that all the collaborative members are already in place."""
        for cm in self.cMem_list:
            self.send_control_data({"type":"confirm connection", "content":"all in place"}, cm['ip'], cm['port'])
    