import time

class Task_Manager():
    """Generate and analyse computing tasks.
    
    Attributes:
        inter_commu (object): Internal communication module.
        local_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
        ip (str): Target ip.
    """
    def __init__(self, local_type, inter_commu, model_manager, local_ip):
        self.inter_commu = inter_commu
        self.model_manager = model_manager
        self.local_type = local_type
        self.ip = local_ip

    def task_send(self, task_type, ip, port, model_id, nm_lst, send_node, range_in=None, padding=None):
        """Send task.
        
        Args:
            task_type (str): task label.
            ip (str): Target ip.
            port (int): target control messages transmission port.
            model_id (int): model id.
            nm_lst (list): A list contains operator names of model.
            send_node (str): The target node for data transmission.
            range_in (list): input range contains input channel, width, and height.
            padding (list): A list of padding parameters.
        """
        if task_type == 'update model':
            content = {
                'task_type': task_type,
                'model_id': model_id,
                'nm_lst': nm_lst,
                'range_in': range_in,
                'send_node': send_node
            }
        elif task_type == 'update submodel':
            content = {
                'task_type': task_type,
                'submodel_id': model_id,
                'nm_lst': nm_lst,
                'padding': padding,
                'send_node': send_node,
                'range_in': range_in
            }

        data = {'type': 'task', 'content': content}
        self.inter_commu.send_control_data(data, ip, port)
    
    def task_parse(self):
        """Analyse received task messages.
        
        Including update model, update submodel, update cluster ability, update configurations, etc.
        """
        while True:
            while len(self.inter_commu.task_queue) == 0:
                time.sleep(0.001)
            task = self.inter_commu.get_task()
            task_type = task['task_type']
            if task_type == 'update model':
                self.inter_commu.task_waiting.set()
                model_id = task['model_id']
                nm_lst = task['nm_lst']
                range_in = task['range_in']
                send_node = task['send_node']
                self.inter_commu.task_waiting.set()
                if self.inter_commu.out_range_lock.acquire():
                    self.inter_commu.out_range.append(None)
                    self.inter_commu.out_range_lock.release()
                self.inter_commu.add_model_id(model_id)
                self.inter_commu.add_model(nm_lst)
                self.inter_commu.add_model_submodel([])
                self.inter_commu.update_rangein(range_in)
                self.inter_commu.add_send_node(send_node)
                if self.inter_commu.inter_range_lock.acquire():
                    self.inter_commu.inter_range.append([])
                    self.inter_commu.inter_range_lock.release()
                self.inter_commu.rescheduling.set()
                # task generate wait
                
                print("Get a new model.")
            
            elif task_type == 'update submodel':
                submodel_id = task['submodel_id']
                nm_lst = task['nm_lst']
                padding = task['padding']
                send_node = task['send_node']
                range_in = task['range_in']
                if submodel_id == 0:
                    self.inter_commu.add_submodel([nm_lst, padding,'test'])
                else:
                    self.inter_commu.add_submodel([nm_lst, padding])
                self.inter_commu.add_submodel_id(submodel_id)
                if not range_in == None:
                    self.inter_commu.update_rangein(range_in)
                
                if self.local_type[-1] == 'H':
                    self.inter_commu.add_model_submodel(submodel_id)
                elif self.local_type[-1] == 'M':
                    self.inter_commu.add_send_node(send_node)
            elif task_type == 'cluster ability':
                ch_ip = task['ip']
                ch_ability = task['ability']
                i = 0
                for ch in self.inter_commu.cHead_list:
                    if ch['ip'] == ch_ip:
                        content = {
                                'task_type': 'confirm ability',
                                'ip': self.ip
                            }
                        data = {'type': 'task', 'content': content}
                        self.inter_commu.send_control_data(data, ch['ip'], ch['port'])
                        break
                    i += 1
                self.inter_commu.clusters_ability[i] = ch_ability
                print("Get ability of ", ch_ip)
            elif task_type == 'config':
                print("Get a config")
                i = 0
                for cm in self.inter_commu.cMem_list:
                    if cm['ip'] == task['ip']:
                        content = {
                                'task_type': 'confirm config'
                            }
                        data = {'type': 'task', 'content': content}
                        self.inter_commu.send_control_data(data, cm['ip'], cm['port'])
                        break
                    i += 1
                if self.inter_commu.cm_config_lock.acquire():
                    self.inter_commu.cm_config[i] = task['content']
                    self.inter_commu.cm_config_lock.release()
            elif task_type == 'confirm ability':
                ch_ip = task['ip']
                for i, ch in enumerate(self.inter_commu.ch_avi):
                    if ch[0] == ch_ip:
                        self.inter_commu.abi_cfm[i] = True
            elif task_type == 'confirm config':
                self.inter_commu.config_cfm.set()
