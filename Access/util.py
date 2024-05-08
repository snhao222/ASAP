from threading import Thread
import time
from .server import SSDPServer
from .client import SSDPClient

class search_declare():
    """Generate SSDP server or client.

    Attributes:
        inter_commu (object): Internal communication module.
        addr_manager(object): Address manager module.
        local type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
        port (int): Control messages transmission port.
        s_ip_list (list): A list contains ip of received access devices.
        s_port_list (list): A list contains ports of received access devices.
        s_nm_list (list): A list contains local labels of received devices.
        c_ip_list (list): A list contains ip of response devices.
        c_port_list (list): A list contains ports of response devices.
        c_nm_list (list): A list contains local labels of response devices.
    """
    def __init__(self, inter_commu, addr_manager, local_type, port=''):
        if local_type == 'Terminal':
            self.nm = 'Ground Station'
        else:
            self.nm = 'Cluster '+local_type[-2]+ (' Head' if local_type[-1]=='H' else ' Member' if local_type[-1]=='M' else '')
        self.port = port
        self.s_ip_list = []
        self.s_port_list = []
        self.s_nm_list = []
        self.c_ip_list = []
        self.c_port_list = []
        self.c_nm_list = []

        if local_type[-1] == 'H':
            Thread(target = self.server, args = (inter_commu, addr_manager, local_type)).start()
            Thread(target = self.client, args = (inter_commu, addr_manager, local_type)).start()
            while True:
                pass
        elif local_type[-1] == 'M' or local_type == 'Terminal':
            self.client(inter_commu, addr_manager, local_type)
        else:
            print("Wrong local device!")

    def server(self, inter_commu, addr_manager, local_type):
        """Generate an SSDP server"""
        server = SSDPServer(inter_commu, addr_manager, self.nm, local_type=local_type, ip_list=self.s_ip_list, c_ip_list=self.c_ip_list, nm_list=self.s_nm_list, port_list=self.s_port_list, s_port = self.port)
        server.serve_forever()
    
    def client(self, inter_commu, addr_manager, local_type):
        """Generate an SSDP client.

        Including M-SEARCH sending and response information parsing.
        """
        client = SSDPClient(self.nm, self.port)
        while True:
            if inter_commu.start_event.is_set():
                time.sleep(0.1)
                continue
            else:
                devices = client.m_search(local_type)
                if not devices == []:
                    for device in devices:
                        ip = device.get("location").split(":")[0]
                        port = int(device.get("location").split(":")[1])
                        c_nm = device.get("usn")
                        alive_port = int(device.get("alive-port"))
                        if local_type[-1] == 'M':
                            data_port = int(device.get("location").split(":")[2])
                            inter_commu.data_port.append(data_port)
                        if (ip not in self.c_ip_list):
                            addr_manager.add_address('CH', c_nm, ip, port, alive_port_s=alive_port)
                            self.c_nm_list.append(c_nm)
                            self.c_ip_list.append(ip)
                            self.c_port_list.append(port)
                            print("Connected Cluster Head: ", inter_commu.cHead_list)
                            print("_"*100)
                        else:
                            time.sleep(0.1)
                            continue