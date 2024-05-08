import threading
from threading import Thread
import pickle
import gzip
import time

class send():
    """Send input data or results to other devices."""
    def __init__(self, data_communication, result_communication, inter_commu, local_type):
        thread = []
        self.socket_lock = threading.Lock()
        if local_type[-1] == 'H':
            self.cm_lst = []
            self.dport = []
            for cm in inter_commu.cMem_list:
                self.cm_lst.append(cm['ip'])
                self.dport.append(cm['data_port'])
            thread.append(Thread(target = self.send2cm, args = (data_communication, inter_commu)))

        elif local_type[-1] == 'M':
            thread.append(Thread(target = self.send2ch, args = (data_communication, inter_commu)))

        thread.append(Thread(target = self.send_result, args = (result_communication, inter_commu)))
        for t in thread:
            t.start()
        while True:
            time.sleep(100)

    def send2ch(self, data_communication, inter_commu):
        """Send input data to cluster head.
        
        Args:
            data_communication (object): A communication object for data transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            if len(inter_commu.send_queue) > 0:
                if inter_commu.send_queue_lock.acquire():
                    sdata = inter_commu.send_queue.pop(0)
                    inter_commu.send_queue_lock.release()
                data = sdata[0]
                ip = sdata[1]
                data_communication.send_message(data, ip, inter_commu.data_port[0])
            else:
                time.sleep(0.001)

    def send2cm(self, data_communication, inter_commu):
        """Send input data to cluster members.
        
        Args:
            data_communication (object): A communication object for data transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            if len(inter_commu.send_queue) > 0:
                if inter_commu.send_queue_lock.acquire():
                    sdata = inter_commu.send_queue.pop(0)
                    inter_commu.send_queue_lock.release()
                data = sdata[0]
                ip = sdata[1]
                port = self.dport[self.cm_lst.index(ip)]
                data_communication.send_message(data, ip, port)
            else:
                time.sleep(0.001)

    def send_result(self, result_communication, inter_commu):
        """Send results to other devices.
        
        Args:
            result_communication (object): A communication object for results transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            if len(inter_commu.results_queue) > 0:
                if inter_commu.results_queue_lock.acquire():
                    sdata = inter_commu.results_queue.pop(0)
                    inter_commu.results_queue_lock.release()
                data = sdata[0]
                ip = sdata[1]
                port = sdata[2]
                result_communication.send_message(data, ip, port)
            else:
                time.sleep(0.001)


class recv():
    """Receive input data or results."""
    def __init__(self, Communication, result_communication, inter_commu, local_type):
        thread = []
        if local_type[-1] == 'H':
            for cm in inter_commu.cMem_list:
                thread.append(Thread(target = self.receive_data, args = (Communication(inter_commu, cm['data_port']), inter_commu, )))
        elif local_type[-1] == 'M':
            thread.append(Thread(target = self.receive_data, args = (Communication(inter_commu, inter_commu.data_port[0]), inter_commu, )))

        thread.append(Thread(target = self.receive_result, args = (result_communication, inter_commu, )))
        for t in thread:
            t.start()
        while True:
            time.sleep(100)
    
    def receive_data(self, data_communication, inter_commu):
        """Receive input data or partial results.
        
        Args:
            data_communication (object): A communication object for data transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            while len(inter_commu.receive_queue) > 30:
                time.sleep(0.001)
            raw_data = data_communication.get_message()
            try:
                data = gzip.decompress(raw_data)
                data = pickle.loads(data)
                data['content'] = data['content'].dequantize()
            except:
                print('data error')
                continue
            try:
                if data["type"] == "data":
                    while len(inter_commu.data_queue) >= 15:
                        time.sleep(0.001)
                    if inter_commu.data_queue_lock.acquire():
                        inter_commu.data_queue.append(data)
                        inter_commu.data_queue_lock.release()

                elif data["type"] == "partial":
                    while len(inter_commu.partial_queue) >= 31:
                        time.sleep(0.001)
                    if inter_commu.partial_queue_lock.acquire():
                        inter_commu.partial_queue.append(data)
                        inter_commu.partial_queue_lock.release()

                elif data["type"] == "test":
                    pass
            except:
                print("Data Formate Error.")
    
    def receive_result(self, result_communication, inter_commu):
        """Receive results from cluster heads.
        
        Args:
            result_communication (object): A communication object for results transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            while len(inter_commu.receive_result_queue) > 15:
                time.sleep(0.001)
            raw_data = result_communication.get_message()
            try:
                data = gzip.decompress(raw_data)
                data = pickle.loads(data)
                data['content'] = data['content'].dequantize()
            except:
                print('data error')
                continue
            try:
                if data["type"] == "result":
                    while len(inter_commu.data_queue) >= 15:
                        time.sleep(0.001)
                    if inter_commu.data_queue_lock.acquire():
                        inter_commu.data_queue.append(data)
                        inter_commu.data_queue_lock.release()

            except:
                print("Data Formate Error.")

class control_data():
    """Send and receive control messages."""
    def __init__(self, communication, inter_commu):
        thread = []
        thread.append(Thread(target = self.send_control_data, args = (communication, inter_commu)))
        thread.append(Thread(target = self.receive_control_data, args = (communication, inter_commu, )))
        for t in thread:
            t.start()
        while True:
            time.sleep(100)
    
    def send_control_data(self, communication, inter_commu):
        """Send control messages.
        
        Args:
            communication (object): A communication object for control messages transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            if len(inter_commu.send_control_queue) > 0:
                if inter_commu.send_control_queue_lock.acquire():
                    sdata = inter_commu.send_control_queue.pop(0)
                    inter_commu.send_control_queue_lock.release()
                data = sdata[0]
                ip = sdata[1]
                port = sdata[2]
                communication.send_message(data, ip, port)
    
    def receive_control_data(self, communication, inter_commu):
        """Receive control messages.
        
        Args:
            communication (object): A communication object for control messages transmission.
            inter_commu (object): Internal communication module.
        """
        while True:
            while len(inter_commu.receive_control_queue) > 15:
                time.sleep(0.001)
            raw_data = communication.get_message()
            try:
                data = pickle.loads(raw_data)
            except:
                print('control data error')
                continue
            try:
                if data["type"] == "confirm connection":
                    inter_commu.recv_confirm_event.set()
                elif data["type"] == "task":
                    if inter_commu.task_queue_lock.acquire():
                        inter_commu.task_queue.append(data["content"])
                        inter_commu.task_queue_lock.release()
            except:
                print("Data Formate Error.")
