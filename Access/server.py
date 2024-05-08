# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import socket
import struct
from .protocol import create_notify_payload
from .http_helper import parse_headers

ipv4_multicast_ip = "239.255.255.250"

logger = logging.getLogger("ssdpy.server")

class SSDPServer(object):
    """A server that can listen to SSDP M-SEARCH requests and responds with
    appropriate NOTIFY packets when the ST matches its device_type.

    Attributes:
        inter_commu (object): Internal communication module.
        addr_manager(object): Address manager module.
        usn (str): A unique service name, which identifies your service.
        local_type (str): Local label, including characteristic in the swarm (i.e., cluster head or
                          cluster member) and corresponding identifier.
        location (str): Canonical URL of the service.
        max_age (int): The maximum time, in seconds, for clients to cache notifications.
        _broadcast_ip (str): Ipv4 multicast ip.
        _address (tuple): (ipv4 multicast ip, ipv4 multicast port).
        sock: UDP socket.
        ip_list (list): A list contains ip of discovered devices.
        port_list (list): A list contains communication ports of discovered devices.
        nm_list (list): A list contains local labels of discovered devices.
        alive_port (int): Connection detection ports for access devices.
        local_ip (str): Local ip.
        port (int, optional): Port to listen on. SSDP works on port 1900, which is the default value here.
        address (str, optional): A specific address to bind to. This is required when using IPv6, since
                                 you will have a link-local IP address in addition to at least one actual IP address.
    """

    def __init__(
        self,
        inter_commu,
        addr_manager,
        usn,
        ip_list,
        port_list,
        c_ip_list,
        nm_list,
        local_type,
        port=1900,
        address=None,
        max_age=None,
        location=None,
        s_port = 9000,
        alive_port = 8888
    ):

        self.inter_commu = inter_commu
        self.addr_manager = addr_manager
        #self.stopped = False
        self.usn = usn
        self.local_type = local_type
        self.location = location
        self.max_age = max_age

        self._af_type = socket.AF_INET
        self._broadcast_ip = ipv4_multicast_ip
        self._address = (self._broadcast_ip, port)
        bind_address = "0.0.0.0"

        self.sock = socket.socket(self._af_type, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Subscribe to multicast address
        mreq = socket.inet_aton(self._broadcast_ip)
        if address is not None:
            mreq += socket.inet_aton(address)
        else:
            mreq += struct.pack(b"@I", socket.INADDR_ANY)
        self.sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq,
        )
        # Allow multicasts on loopback devices (necessary for testing)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        
        self.sock.bind((bind_address, port))

        self.ip_list = ip_list
        self.c_ip_list = c_ip_list
        self.port_list = port_list
        self.port = s_port
        self.nm_list = nm_list
        self.alive_port = alive_port
        self.data_port = 6600

        ip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip_socket.connect(('8.8.8.8',80))
        self.local_ip = ip_socket.getsockname()[0]

    def on_recv(self, data, address):
        logger.debug("Received packet from {}: {}".format(address, data))
        try:
            headers = parse_headers(data)
        except ValueError:
            # Not an SSDP M-SEARCH; ignore.
            logger.debug("NOT M-SEARCH - SKIPPING")
            pass
        if data.startswith(b"M-SEARCH") and ((headers.get("st")[-1]=='H' and headers.get("st")[-2]!=self.local_type[-2]) or\
                                              (headers.get("st")[-1]=='M' and headers.get("st")[-2]==self.local_type[-2])) and\
                                                  (address[0] not in self.ip_list) and (address[0] != self.local_ip):
            
            logger.info("Received qualifying M-SEARCH from {}".format(address))
            logger.debug("M-SEARCH data: {}".format(headers))
            if headers.get("st")[-1]=='M':
                notify = create_notify_payload(
                host=self._broadcast_ip,
                nt=self.local_type,
                usn=self.usn,
                location=str(self.local_ip)+ ":"+ str(self.port)+ ":"+ str(self.data_port),
                max_age=self.max_age,
                alive_port=self.alive_port
                )
                self.data_port += 1
            else:
                notify = create_notify_payload(
                    host=self._broadcast_ip,
                    nt=self.local_type,
                    usn=self.usn,
                    location=str(self.local_ip)+ ":"+ str(self.port),
                    max_age=self.max_age,
                    alive_port=self.alive_port
                )

            if headers.get("st")[-1] == 'H':
                self.addr_manager.add_address('CH', headers.get("nm"), address[0], int(headers.get("host").split(":")[1]), self.alive_port)
                if self.inter_commu.ch_avi_lock.acquire():
                    self.inter_commu.ch_avi.append([address[0], int(headers.get("host").split(":")[1])])
                    self.inter_commu.ch_avi_lock.release()
                print("Connected Cluster Head: ", self.inter_commu.cHead_list)
                print("_"*100)

            elif headers.get("st")[-1] == 'M':
                self.addr_manager.add_address('CM', headers.get("nm"), address[0], int(headers.get("host").split(":")[1]), self.alive_port, data_port=(self.data_port-1))
                if self.inter_commu.cm_avi_lock.acquire():
                    self.inter_commu.cm_avi.append([address[0], int(headers.get("host").split(":")[1])])
                    self.inter_commu.cm_avi_lock.release()
                if self.inter_commu.cm_config_lock.acquire():
                    self.inter_commu.cm_config.append(None)
                    self.inter_commu.cm_config_lock.release()
                print("Connected Cluster Mem.: ", self.inter_commu.cMem_list)
                print("_"*100)
            elif headers.get("st")[-1] == 'Terminal':
                self.addr_manager.add_address('Terminal', headers.get("nm"), address[0], int(headers.get("host").split(":")[1]))
                print("Connected Ground Station: ", address[0])
            # alive port
            self.alive_port += 1

            self.nm_list.append(headers.get("nm"))
            self.ip_list.append(address[0])
            self.port_list.append(headers.get("host").split(":")[1])
            #print(self.ip_list, self.port_list, self.nm_list)

            logger.debug("Created NOTIFY: {}".format(notify))
            try:
                self.sock.sendto(notify, address)
                self.sock.sendto(notify, address)
                self.sock.sendto(notify, address)
            except OSError as e:
                # Most commonly: We received a multicast from an IP not in our subnet
                logger.debug("Unable to send NOTIFY to {}: {}".format(address, e))

    def serve_forever(self):
        """Start listening for M-SEARCH discovery attempts and answer any that refers to
        our ``device_type`` or to ``ssdp:all``. This will block execution until an exception occurs.
        """
        logger.info("Listening forever")
        try:
            #while not self.stopped:
            while not self.inter_commu.start_event.is_set():
                data, address = self.sock.recvfrom(1024)
                self.on_recv(data, address)
        except Exception:
            self.sock.close()
            raise