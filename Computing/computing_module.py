import torch
from threading import Thread
import time

class Computing():
    """Perform DL inference.
    
    Attributes:
        inter_commu (object): Internal communication module.
        model_manager (object): Model manager module.
    """
    def __init__(self, inter_commu, model_manager, task_type):
        self.inter_commu = inter_commu
        self.device = torch.device('cuda')
        self.submodel = []
        self.model_manager = model_manager
        self.task_type = task_type
        Thread(target = self.prepare_model).start()

        while True:
            while len(inter_commu.candidate_queue) == 0:
                time.sleep(0.001)
            data = self.inter_commu.get_candidate()
            data_id = data[0]
            seq = data[1]
            submodel_id = data[2]
            input_data = data[3]

            while submodel_id > len(self.submodel)-1:
                time.sleep(0.001)
            result = self.computing(submodel_id, input_data)
            if self.inter_commu.partial_queue_lock.acquire():
                self.inter_commu.partial_queue.append({
                    'type': 'partial',
                    'data_id': data_id,
                    'seq': seq,
                    'submodel_id': submodel_id,
                    'content': result.to("cpu")
                })
                self.inter_commu.partial_queue_lock.release()

    def computing(self, submodel_id, data):
        """DL inference.
        
        Args:
            submodel_id (int): Submodel id.
            data (array): Data to be computed.
        
        Returns:
            Computed results.
        """
        with torch.no_grad():
            while submodel_id not in self.inter_commu.submodel_id:
                time.sleep(0.1)
            result = self.submodel[self.inter_commu.submodel_id.index(submodel_id)](data.to(self.device))
            torch.cuda.empty_cache()
            return result
    
    def prepare_model(self):
        """Generate a model.
        
        Containing model construction and model padding.
        """
        while True:
            while len(self.inter_commu.submodel) == 0:
                time.sleep(0.01)
            if self.inter_commu.submodel_lock.acquire():
                submodel_para = self.inter_commu.submodel.pop(0)
                self.inter_commu.submodel_lock.release()
            if len(submodel_para) == 3:
                submodel = self.model_manager.test_model
                self.model_manager.model_padding_test(submodel, submodel_para[0], submodel_para[1])
            else:
                submodel = self.model_manager.model_gen(submodel_para[0])
                self.model_manager.model_padding(submodel, submodel_para[0], submodel_para[1])
            
            submodel = submodel.to(self.device)
            print("Task Model:", submodel)
            x_act = torch.randn(1, self.inter_commu.range_in[0][0], self.inter_commu.range_in[0][1], self.inter_commu.range_in[0][2], dtype=torch.float).to(self.device)
            with torch.no_grad():
                for _ in range(2):
                    submodel(x_act)
            del x_act
            torch.cuda.empty_cache()
            self.submodel.append(submodel)
            self.inter_commu.task_waiting.clear()

    def warm_up(self):
        """Warm up device with several rounds of computing."""
        model = self.model_manager.model_gen(self.model_manager.nm_lst[0:4]).to(self.device)
        x = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
        with torch.no_grad():
            for _ in range (50):
                model(x)
        del model
        del x
        torch.cuda.empty_cache()
        print("[Warm up over]")
        self.inter_commu.warm_up.set()
