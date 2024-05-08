import os
import sys
import pickle
import threading
from threading import Thread
import time
import yaml
import struct
import socket
import random

class performance_declaration():
    """Performance declaration.
    
    Alive information declaration and send rate changes.
    
    Attributes:
        task_owner (bool): "True" when the local device is the task owner.
        communication (object): External communication module.
        inter_commu (object): Internal communication module.
        local_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
        parapath (str): Predictor configuration file path.
        fitpath (str): File path of basic latency pattern.
        state_manager (object): State manage module.
        rate (list): A list of communication rates between nodes.
    """
    def __init__(self, task_owner, communication, inter_commu, local_type, parapath, fitpath, Communication, state_manager):
        self.inter_commu = inter_commu
        self.communication = communication
        self.local_type = local_type
        self.parapath = parapath
        self.fitpath = fitpath
        self.Communication = Communication
        self.state_manager = state_manager
        self.rate = [0]*len(self.inter_commu.rate)
        self.task_owner = task_owner
        self.keep_alive()

    def keep_alive(self):
        """Main function of performance declaration module.
        
        Send predictor configurations and basic latency patterns, and initialize rate_manage function and send_alive_perfo function. 
        """
        threads = []
        if not self.task_owner:
            for ch in self.inter_commu.cHead_list:
                threads.append(Thread(target = self.send_alive_perfo, args = (ch['ip'], ch['alive-port_s'])))
        if self.local_type[-1] == 'M':
            threads.append(Thread(target = self.rate_manage))
        
        for t in threads:
            t.start()
        if self.local_type[-1] == 'M':
            parapath = os.path.join(os.getcwd(), self.parapath)
            with open(parapath, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            fitpath = os.path.join(os.getcwd(), self.fitpath)
            with open(fitpath, 'r', encoding='utf-8') as f:
                fitting = yaml.load(f.read(), Loader=yaml.FullLoader)
            content = {'task_type':'config', 'ip':self.communication.ip, 'content':[configs, fitting]}
            data = {'type': 'task', 'content': content}
            sleeptime=random.randint(0, 5)
            time.sleep(sleeptime)
            self.inter_commu.send_control_data(data, self.inter_commu.cHead_list[0]['ip'], self.inter_commu.cHead_list[0]['port'])
            time.sleep(5)
            while not self.inter_commu.config_cfm.is_set():
                self.inter_commu.send_control_data(data, self.inter_commu.cHead_list[0]['ip'], self.inter_commu.cHead_list[0]['port'])
                time.sleep(5)
        while True:
            time.sleep(100)

    def rate_manage(self):
        """Update communication rate when value changes."""
        rate_0 = [0]*len(self.inter_commu.rate)
        timer = [5]*len(self.inter_commu.rate)
        while True:
            for i, rate in enumerate(self.inter_commu.rate):
                if rate_0[i] == 0:
                    self.rate[i] = rate
                    rate_0[i] = rate
                elif abs(rate-rate_0[i])/rate_0[i] > 0.3:
                    timer[i] = timer[i]-1
                    if timer[i] < 0:
                        self.rate[i] = rate
                        timer[i] = 5
                        rate_0[i] = rate
            time.sleep(2)

    def send_alive_perfo(self, ip, port):
        """Send alive declaration information.
        
        Args:
            ip (str): Target ip.
            port (int): Target alive declaration detection port.
        """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            if self.rate[self.inter_commu.ch_ip.index(ip)] != 0:
                vary_rate = struct.pack('i', self.rate[self.inter_commu.ch_ip.index(ip)])
                self.rate[self.inter_commu.ch_ip.index(ip)] = 0
                client_socket.sendto(vary_rate, (ip, port))
            else:
                client_socket.sendto(struct.pack('i', 123), (ip, port))
            time.sleep(2)

class connection_detector():
    """Connection detector.
    
    Receive alive declaration information and detect unavailable node.

    Attributes:
        communication (object): External communication module.
        inter_commu (object): Internal communication module.
        task_owner (bool): "True" when the local node is the task owner.
        ip_list (list): A list contains ips of collaborative nodes.
    """
    def __init__(self, communication, inter_commu, state_manager, task_owner):
        self.inter_commu = inter_commu
        self.ip_list = []
        self.status = []
        threads = []
        self.status_lock = threading.Lock()
        self.recv_lock = threading.Lock()
        if task_owner:
            for ch in self.inter_commu.cHead_list:
                self.ip_list.append(ch['ip'])
                if self.status_lock.acquire():
                    self.status.append(False)
                    self.status_lock.release()
                threads.append(Thread(target = self.recv_declaration, args = (communication.ip, ch['alive-port_l'], ch['ip'], ch['port'], 'CH')))
        for cm in self.inter_commu.cMem_list:
            self.ip_list.append(cm['ip'])
            if self.status_lock.acquire():
                self.status.append(False)
                self.status_lock.release()
            threads.append(Thread(target = self.recv_declaration, args = (communication.ip, cm['alive-port_l'], cm['ip'], cm['port'], 'CM')))
        for t in threads:
            t.start()
        while True:
            time.sleep(100)

    def recv_declaration(self, ip, port, cm_ip, cm_port, device_type):
        """Manage alive situation of one node and update communication rate between nodes.
        
        Args:
            ip (str): Local ip.
            port (int): Alive declaration detection port.
            cm_ip (str): Ip of target node.
            cm_port (int): Control messages transmission port of target node.
            device_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                               cluster member) and corresponding identifier.
        """
        master_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)     
        master_socket.bind((ip, port))
        data, a = master_socket.recvfrom(100)
        master_socket.settimeout(5)

        while True:
            while True:
                try:
                    data, a = master_socket.recvfrom(100)
                    if struct.unpack('i',data)[0] != 123:
                        self.inter_commu.cm_rate[self.inter_commu.cm_ip.index(cm_ip)] = struct.unpack('i',data)[0]
                        print("{} rate: {}" .format(cm_ip, struct.unpack('i',data)[0]))
                except:
                    try:
                        #master_socket.settimeout(2)
                        data, a = master_socket.recvfrom(100)
                        if struct.unpack('i',data)[0] != 123:
                            self.inter_commu.cm_rate[self.inter_commu.cm_ip.index(cm_ip)] = struct.unpack('i',data)[0]
                            print("{} rate: {}" .format(cm_ip, struct.unpack('i',data)[0]))
                    except:
                        print("UNAVAILABLE:", cm_ip)
                        if device_type[-1] == 'H':
                            if self.inter_commu.ch_avi_lock.acquire():
                                self.inter_commu.ch_avi.pop(self.inter_commu.ch_avi.index([cm_ip, cm_port]))
                                self.inter_commu.ch_avi_lock.release()
                            self.inter_commu.ch_unavi_event.set()
                        elif device_type[-1] == 'M':
                            if self.inter_commu.cm_avi_lock.acquire():
                                self.inter_commu.cm_avi.pop(self.inter_commu.cm_avi.index([cm_ip, cm_port]))
                                self.inter_commu.cm_avi_lock.release()
                            self.inter_commu.cm_unavi_event.set()
                        break
            master_socket.settimeout(None)
            master_socket.recvfrom(100)
            master_socket.settimeout(5)
            print(cm_ip,'Connection Recovery.')
            if device_type[-1] == 'H':
                if self.inter_commu.ch_avi_lock.acquire():
                    self.inter_commu.ch_avi.append([cm_ip, cm_port])
                    self.inter_commu.ch_avi_lock.release()
                self.inter_commu.ch_recover_event.set()
            elif device_type[-1] == 'M':
                if self.inter_commu.cm_avi_lock.acquire():
                    self.inter_commu.cm_avi.append([cm_ip, cm_port])
                    self.inter_commu.cm_avi_lock.release()
                self.inter_commu.cm_recover_event.set()



    def alive_timer(self, ip, port, device_type):
        timer_thre = 5
        timer = timer_thre
        while not self.status[self.ip_list.index(ip)]:
            time.sleep(0.001)
        while True:
            if timer == 0:
                '''
                if self.inter_commu.unavailable_lock.acquire():
                    self.inter_commu.unavailable.append(ip)
                    self.inter_commu.unavailable_lock.release()
                '''
                print("UNAVAILABLE:", ip)
                if device_type[-1] == 'H':
                    if self.inter_commu.ch_avi_lock.acquire():
                        self.inter_commu.ch_avi.pop(self.inter_commu.ch_avi.index([ip, port]))
                        self.inter_commu.ch_avi_lock.release()
                    self.inter_commu.ch_unavi_event.set()
                elif device_type[-1] == 'M':
                    if self.inter_commu.cm_avi_lock.acquire():
                        self.inter_commu.cm_avi.pop(self.inter_commu.cm_avi.index([ip, port]))
                        self.inter_commu.cm_avi_lock.release()
                    self.inter_commu.cm_unavi_event.set()
                while not self.status[self.ip_list.index(ip)]:
                    time.sleep(0.1)
                    timer = timer_thre
                if self.status_lock.acquire():
                    self.status[self.ip_list.index(ip)] = False
                    self.status_lock.release()
                    print(ip,'Connection Recovery.')
                    if device_type[-1] == 'H':
                        if self.inter_commu.ch_avi_lock.acquire():
                            self.inter_commu.ch_avi.append([ip, port])
                            self.inter_commu.ch_avi_lock.release()
                        self.inter_commu.ch_recover_event.set()
                    elif device_type[-1] == 'M':
                        if self.inter_commu.cm_avi_lock.acquire():
                            self.inter_commu.cm_avi.append([ip, port])
                            self.inter_commu.cm_avi_lock.release()
                        self.inter_commu.cm_recover_event.set()
            elif self.status[self.ip_list.index(ip)]:
                timer = timer_thre
                if self.status_lock.acquire():
                    self.status[self.ip_list.index(ip)] = False
                    self.status_lock.release()
            else:
                timer -= 1
            time.sleep(1)
