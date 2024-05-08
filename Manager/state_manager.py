
class State_manager():
    """Manage state of collaborative nodes.
    
    Args:
        inter_commu (object): Internal communication module.
        ch_list (list): A list of cluster heads' ip.
        cm_list (list): A list of cluster members' ip.
    """
    def __init__(self, inter_commu):
        self.ch_list = []
        self.cm_list = []
        self.inter_commu = inter_commu
        for ch in inter_commu.cHead_list:
            self.ch_list.append(ch['ip'])
            inter_commu.ch_perfo.append(None)
            inter_commu.clusters_ability.append(None)
            self.inter_commu.ch_ip.append(ch['ip'])
            self.inter_commu.rate.append(0)
        for cm in inter_commu.cMem_list:
            self.cm_list.append(cm['ip'])
            self.inter_commu.cm_ip.append(cm['ip'])
            inter_commu.cm_perfo.append(None)
            self.inter_commu.cm_rate.append(None)
            self.inter_commu.cm_time.append(0)
            self.inter_commu.cm_rate_0.append(None)
    
    def update(self, perfo_type, ip, frq, mem_avi, task_num, rate):
        if perfo_type[-1] == 'H':
            self.inter_commu.ch_perfo[self.ch_list.index(ip)]={
                'ip': ip,
                'frq': frq,
                'mem_avi': mem_avi,
                'task_num': task_num,
                'rate': rate
            }
        elif perfo_type[-1] == 'M':
            self.inter_commu.cm_perfo[self.cm_list.index(ip)]={
                'ip': ip,
                'frq': frq,
                'mem_avi': mem_avi,
                'task_num': task_num,
                'rate': rate
            }
        print('CH:',self.ch_list)
        print(self.inter_commu.ch_perfo)
        print('CM', self.cm_list)
        print(self.inter_commu.cm_perfo)
    
    def check(self, perfo_type, ip):
        if perfo_type == 'CH':
            check_data = self.inter_commu.ch_perfo[self.ch_list.index(ip)]
        elif perfo_type == 'CM':
            check_data = self.inter_commu.cm_perfo[self.cm_list.index(ip)]
        
        return (check_data['frq'], check_data['mem_avi'], check_data['task_num'], check_data['rate'])