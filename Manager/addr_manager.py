import threading

class Addr_manager():
    """Manage address information of cluster heads and cluster members.
    
    Attributes:
        ch_list (list): A list contains the address information of cluster heads.
        cm_list (list): A list contains the address information of cluster members.
    """
#---------------------------------------------------------------------
#  Input: user_name, ip, port
# Output: a list of dictionary containing user_name, ip and port
#---------------------------------------------------------------------
    def __init__(self, inter_commu):
        self.inter_commu = inter_commu
        self.ch_list = []
        self.ch_lock = threading.Lock()
        self.cm_list = []
        self.cm_lock = threading.Lock()
        self.ground_station = []
        self.gs_lock = threading.Lock()
    
    def add_address(self, dev_type, nm, ip, port, alive_port_l=None, alive_port_s=None, data_port =None):
        """Add address information.
        
        Args:
            dev_type (str): Local label, including characteristic in the swarm (i.e., cluster head or cluster member).
            nm (str): Node label.
            ip (str): Node ip.
            port (int): Control messages transmission port.
            alive_port_l (int): Alive declaration detection port.
            alive_port_s (int): Alive declaration send port.
            data_port (int): Data transmission port.
        """
        if dev_type == "CH":
            if ip not in self.ch_list:
                if self.inter_commu.cHead_lock.acquire():
                    self.inter_commu.cHead_list.append(
                        {
                            'nm': nm,
                            'ip': ip,
                            'port': port,
                            'alive-port_l': alive_port_l,
                            'alive-port_s': alive_port_s
                        }
                    )
                    self.inter_commu.cHead_lock.release()
                if self.ch_lock.acquire():
                    self.ch_list.append(ip)
                    self.ch_lock.release()
            elif alive_port_l == None:
                raw_data = self.inter_commu.cHead_list[self.ch_list.index(ip)]
                if self.inter_commu.cHead_lock.acquire():
                    self.inter_commu.cHead_list[self.ch_list.index(ip)] = {
                        'nm': raw_data['nm'],
                        'ip': raw_data['ip'],
                        'port': raw_data['port'],
                        'alive-port_l': raw_data['alive-port_l'],
                        'alive-port_s': alive_port_s
                    }
                    self.inter_commu.cHead_lock.release()
            else:
                raw_data = self.inter_commu.cHead_list[self.ch_list.index(ip)]
                if self.inter_commu.cHead_lock.acquire():
                    self.inter_commu.cHead_list[self.ch_list.index(ip)] = {
                        'nm': raw_data['nm'],
                        'ip': raw_data['ip'],
                        'port': raw_data['port'],
                        'alive-port_l': alive_port_l,
                        'alive-port_s': raw_data['alive-port_s']
                    }
                    self.inter_commu.cHead_lock.release()

        elif dev_type == 'CM':
            if ip not in self.cm_list:
                if self.inter_commu.cMem_lock.acquire():
                    self.inter_commu.cMem_list.append(
                        {
                            'nm': nm,
                            'ip': ip,
                            'port': port,
                            'alive-port_l': alive_port_l,
                            'alive-port_s': alive_port_s,
                            'data_port': data_port
                        }
                    )
                    self.inter_commu.cMem_lock.release()
                if self.cm_lock.acquire():
                    self.cm_list.append(ip)
                    self.cm_lock.release()
            elif alive_port_l == None:
                raw_data = self.inter_commu.cMem_list[self.cm_list.index(ip)]
                if self.inter_commu.cMem_lock.acquire():
                    self.inter_commu.cMem_list[self.cm_list.index(ip)] = {
                        'nm': raw_data['nm'],
                        'ip': raw_data['ip'],
                        'port': raw_data['port'],
                        'alive-port_l': raw_data['alive-port_l'],
                        'alive-port_s': alive_port_s,
                        'data_port': raw_data['data_port']
                    }
                    self.inter_commu.cMem_lock.release()
            else:
                raw_data = self.inter_commu.cMem_list[self.cm_list.index(ip)]
                if self.inter_commu.cMem_lock.acquire():
                    self.inter_commu.cMem_list[self.cm_list.index(ip)] = {
                        'nm': raw_data['nm'],
                        'ip': raw_data['ip'],
                        'port': raw_data['port'],
                        'alive-port_l': alive_port_l,
                        'alive-port_s': raw_data['alive-port_s'],
                        'data_port': raw_data['data_port']
                    }
                    self.inter_commu.cMem_lock.release()
        elif dev_type == 'Terminal':
            if ip not in self.ground_station:
                if self.inter_commu.ground_station_lock.acquire():
                    self.inter_commu.ground_station.append(
                        {
                            'nm': nm,
                            'ip': ip,
                            'port': port
                        }
                    )
                    self.inter_commu.ground_station_lock.release()
                if self.gs_lock.acquire():
                    self.ground_station.append(ip)
                    self.gs_lock.release()

    def del_address(self, dev_type, ip):
        """Delete address information from list.
        
        Args:
            dev_type (str): Local label, including characteristic in the swarm (i.e., cluster head or cluster member).
            ip (str): Device ip.
        """
        idx = 0
        if dev_type == "CH":
            if self.inter_commu.cHead_lock.acquire():
                for dict in self.inter_commu.cHead_list:
                    if dict['ip'] == ip:
                        self.inter_commu.cHead_list.pop(idx)
                    idx += 1
                self.inter_commu.cHead_lock.release()
        elif dev_type == "CM":
            if self.inter_commu.cMem_lock.acquire():
                for dict in self.inter_commu.cHead_list:
                    if dict['ip'] == ip:
                        self.inter_commu.cMem_list.pop(idx)
                    idx += 1
                self.inter_commu.cMem_lock.release()
