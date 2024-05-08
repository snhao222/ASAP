__package__ = 'Model'
import torch
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import time
from .util import decom_vgg16, Process, tranform

class Task_generator():
    """Generate computing tasks for the swarm.

    'vgg' ----- VGG16
    'resnet' -- ResNet34, ResNet50, ResNet101, ResNet152
    'darknet' -- DarkNet19
    
    Attributes:
        task_type (str): Model name of task.
        inter_commu (object): Internal communication module.
    """
    def __init__(self, task_type, inter_commu):
        self.data_id = 30
        
        if task_type == 'vgg':
            self.detection_dataset(inter_commu)
        
        elif task_type == 'faster-rcnn':
            self.detection_dataset(inter_commu)
            
        elif 'resnet' in task_type:
            self.detection_dataset(inter_commu)

        elif task_type == 'darknet':
            self.detection_dataset(inter_commu)

        elif task_type == 'alexnet':
            self.detection_dataset(inter_commu)

        elif task_type == 'pod':
            pass

    def data_set(self, inter_commu):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])
        dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=transform, download=True)
        for _, (data, _) in enumerate(dataset):
            
            while inter_commu.task_waiting.is_set() or len(inter_commu.submodel_id)==0 or len(inter_commu.data_queue)>30 or (not inter_commu.cluster_measure.is_set()):
                time.sleep(0.01)
            data = {
                'data_id': self.data_id,
                'model_id': inter_commu.model_id[-1],
                'content': data.unsqueeze(0)
            }
            inter_commu.add_data(data)
            self.data_id += 1
            time.sleep(0.1)
    
    def detection_dataset(self, inter_commu):
        """Get input data from dataset.
        
        Args:
            inter_commu (object): Internal communication module.
        """
        while True:
            for i in range(628):
                while inter_commu.task_waiting.is_set() or len(inter_commu.submodel_id)==0 \
                    or len(inter_commu.data_queue)>15 or (not inter_commu.cluster_measure.is_set())\
                    or (not self.inter_commu.local_measure.is_set()) or (inter_commu.model_id[-1]== [])\
                    or (inter_commu.model_id[-1]== 0):
                    time.sleep(0.1)
                img = './target video/fig'+str(i+1).zfill(3)+'.jpg'
                try:
                    image = Image.open(img)
                except:
                    print('Open Error!')
                    continue
                image, image_shape, input_shape = tranform(image)
                if len(inter_commu.model_id) == len(inter_commu.submodel_id):
                    model_id = inter_commu.model_id[-1]
                elif len(inter_commu.model_id) > len(inter_commu.submodel_id):
                    model_id = inter_commu.model_id[-2]
                data = {
                    'data_id': self.data_id,
                    'model_id': model_id,
                    'content': image
                }
                inter_commu.add_data(data)
                self.data_id += 1
                time.sleep(0.1)