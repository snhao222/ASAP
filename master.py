import sys
import os
import multiprocessing
from threading import Thread
import numpy as np
import sklearn
import time
import argparse
##########  Import Objects  ##########
from Access.util import search_declare
from Communicate.communication import Communication, Inter_commu
from Manager.addr_manager import Addr_manager
from Manager.state_manager import State_manager
from Manager.model_manager import Model_Manager
from Manager.data_manager import Data_Manager
from Manager.task_manager import Task_Manager
from Computing.computing_module import Computing
from Communicate.recv_send import recv, send, control_data
from State.Connection import performance_declaration, connection_detector
from Scheduler.scheduler_module import Scheduler
from task_generator import Task_generator
from Scheduler.latency_predictor import Latency_predictor
##########    Node Info.  ##########
parser = argparse.ArgumentParser(description='Getting start on a single node')
parser.add_argument('-c', '--cluster', type=int, metavar='', help='Cluster number')
parser.add_argument('-r', '--role', type=str, metavar='', help='\'H\' for cluster head, and \'M\' for cluster member')
parser.add_argument('-t', '--task', type=str, metavar='', required=True, help='\'vgg\' for VGG16; \'resnet34\' for ResNet34; \'resnet50\' for ResNet50; \'resnet101\' for ResNet101; \'resnet152\' for ResNet152; \'darknet\' for DarkNet19')
group = parser.add_mutually_exclusive_group()
group.add_argument('-o', '--owner', action='store_true', help='Default \'False\'')
group.add_argument('-e', '--elastic', action='store_true', help='Default \'False\'')
args = parser.parse_args()
local_type = 'C'+str(args.cluster)+args.role
port = 9999
task_owner = args.owner
task_type = args.task
elastic = args.elastic
parafile = './model_data/para.yml'
fittingfile = './model_data/fitting.yml'
######################################

inter_commu = Inter_commu()
addr_manager = Addr_manager(inter_commu)
communication = Communication(inter_commu, port)
data_communication = Communication(inter_commu, port+2)
result_communication = Communication(inter_commu, port+1)
model_manager = Model_Manager(task_type)
task_manager = Task_Manager(local_type, inter_commu, model_manager, communication.ip)


def confirm_conn(inter_commu):
    """Confirm connection.

    Notify all the nodes that the collaborative process is ready and send confirmation to other cluster heads and cluster members.
    
    Args:
        inter_commu (object): Internal communication module.
    """
    subprocess = []
    while True:
        if not inter_commu.start_event.is_set():
            key = input("_____|Press 'Enter' to start.|_____\n")
            if key == '' and not inter_commu.start_event.is_set():
                inter_commu.terminate.set()
                time.sleep(2)
                inter_commu.confirm_all()
                print('All members in place. Computing start!')
                state_manager = State_manager(inter_commu)
                subprocess.append(multiprocessing.Process(target=performance_declaration, name='Performance Declaration ',\
                                         args=(task_owner, communication, inter_commu, local_type, parafile, fittingfile, Communication, state_manager)))
                subprocess.append(multiprocessing.Process(target=connection_detector, name='Connection Detector     ',\
                                         args=(communication, inter_commu, state_manager, task_owner)))
                subprocess.append(multiprocessing.Process(target=Computing, name='Computing Module      ',\
                                         args=(inter_commu, model_manager, task_type)))
                subprocess.append(multiprocessing.Process(target=Data_Manager, name='Data Management Module      ',\
                                         args=(local_type, inter_commu, communication)))
                
                subprocess.append(multiprocessing.Process(target=Scheduler, name='Scheduler      ',\
                                         args=(inter_commu, task_manager, model_manager, task_owner, communication, state_manager, Latency_predictor,\
                                         parafile, fittingfile, Task_generator, task_type, elastic)))

                subprocess.append(multiprocessing.Process(target=recv, name='Communication Module (Receive)  ',\
                                         args=(Communication, result_communication, inter_commu, local_type)))
                subprocess.append(multiprocessing.Process(target=send, name='Communication Module (Send)     ',\
                                         args=(data_communication, result_communication, inter_commu, local_type)))
                
                task_parse = Thread(target = task_manager.task_parse, name='Task Management Module      ')
                task_parse.start()

                for sp in subprocess:
                    sp.start()
                access.terminate()
                access.join()
        else:
            time.sleep(0.1)


def recv_confirm(inter_commu, local_type):
    """Receive confirmation from cluster heads.
    
    Args:
        inter_commu (object): Internal communication module.
        local_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
    """
    subprocess = []
    while True:
        if inter_commu.terminate.is_set():
            break
        if  inter_commu.recv_confirm_event.is_set() and not inter_commu.start_event.is_set():
            inter_commu.recv_confirm_event.clear()
            inter_commu.start_event.set()
            state_manager = State_manager(inter_commu)
            if local_type[-1] == 'H':
                inter_commu.confirm_members()
                
                subprocess.append(multiprocessing.Process(target=connection_detector, name='Connection Detector     ',\
                                     args=(communication, inter_commu, state_manager, task_owner)))
                subprocess.append(multiprocessing.Process(target=Scheduler, name='Scheduler      ',\
                                         args=(inter_commu, task_manager, model_manager, task_owner, communication, state_manager, Latency_predictor,\
                                        parafile, fittingfile, Task_generator, task_type, elastic)))
                subprocess.append(multiprocessing.Process(target=performance_declaration, name='Performance Declaration ',\
                                     args=(task_owner, communication, inter_commu, local_type, parafile, fittingfile, Communication, state_manager)))

            elif local_type[-1] == 'M':
                subprocess.append(multiprocessing.Process(target=performance_declaration, name='Performance Declaration ',\
                                     args=(task_owner, communication, inter_commu, local_type, parafile, fittingfile, Communication, None)))
            subprocess.append(multiprocessing.Process(target=Computing, name='Computing Module      ',\
                                     args=(inter_commu, model_manager, task_type)))
            subprocess.append(multiprocessing.Process(target=Data_Manager, name='Data Management Module      ',\
                                     args=(local_type, inter_commu, communication)))
            subprocess.append(multiprocessing.Process(target=recv, name='Communication Module (Receive)  ',\
                                     args=(Communication, result_communication, inter_commu, local_type)))
            subprocess.append(multiprocessing.Process(target=send, name='Communication Module (Send)     ',\
                                     args=(data_communication, result_communication, inter_commu, local_type)))

            task_parse = Thread(target = task_manager.task_parse, name='Task Management Module      ')
            task_parse.start()

            for sp in subprocess:
                sp.start()
            print('Receive confirmation. All members in place.')
            access.terminate()
            access.join()
        else:
            time.sleep(0.1)

access = multiprocessing.Process(target=search_declare, name='Access Module         ',\
                                         args=(inter_commu, addr_manager, local_type, port))
access.start()

if local_type[-1] != 'M':
    confirm_conn = Thread(target = confirm_conn, args = (inter_commu, ))
    confirm_conn.start()

recv_confirm = Thread(target = recv_confirm, args = (inter_commu, local_type))
recv_confirm.start()

control_d = Thread(target = control_data, name='Communication Module (Control)  ', args = (communication, inter_commu))
control_d.start()

while True:
    time.sleep(100)