# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import socket
from .http_helper import parse_headers
from .protocol import create_msearch_payload

ipv4_multicast_ip = "239.255.255.250"

class SSDPClient(object):
    """Send access information to other devices.

    An M-SEARCH request is sent and the responses are parsed to get connection information.

    Attributes:
        port (int): Control messages transmission port.
        broadcast_ip (str): Ipv4 multicast ip.
        _address (tuple): (ipv4 multicast ip, ipv4 multicast port).
        sock: UDP socket.
        nm (str): Local label, including characteristic in the swarm (i.e., cluster head or
                  cluster member) and corresponding identifier.
        local_ip (str): Local ip.
    """
    def __init__(
        self,
        nm,
        c_port,
        port=1900,
        ttl=2,
        timeout=5,
        *args,
        **kwargs
    ):
        
        self.port = c_port
        af_type = socket.AF_INET
        self.broadcast_ip = ipv4_multicast_ip
        self._address = (self.broadcast_ip, port)
        
        self.sock = socket.socket(af_type, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.settimeout(timeout)
        self.nm = nm

        ip_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip_socket.connect(('8.8.8.8',80))
        self.local_ip = ip_socket.getsockname()[0]

    def send(self, data):
        self.sock.sendto(data, self._address)

    def recv(self):
        try:
            while True:
                data = self.sock.recv(1024)
                yield data
        except socket.timeout:
            pass
        return

    def m_search(self, st="ssdp:all", mx=1):
        """Send an M-SEARCH request and gather responses.

        Args:
            st (str): The Search Target, used to narrow down the responses
                      that should be received. Defaults to "ssdp:all" which
                      should get responses from any SSDP-enabled device.
            mx (int): Maximum wait time (in seconds) that devices are allowed
                      to wait before sending a response. Should be between 1
                      and 5, though this is not enforced in this implementation.
                      Devices will randomly wait for anywhere between 0 and 'mx'
                      seconds in order to avoid flooding the client that sends
                      the M-SEARCH. Increase the value of 'mx' if you expect a
                      large number of devices to answer, in order to avoid losing responses.

        Returns:
            A list of all discovered SSDP services. Each service is represented
            by a dict, with the keys being the lowercase equivalents of the response headers.
        """
        host = "{}:{}".format(self.local_ip, self.port)
        data = create_msearch_payload(host, st, self.nm, mx)
        self.send(data)
        responses = [x for x in self.recv()]
        parsed_responses = []
        for response in responses:
            try:
                headers = parse_headers(response)
                parsed_responses.append(headers)
            except ValueError:
                # Invalid response, do nothing.
                # TODO: Log dropped responses
                pass
        return parsed_responses


def discover():
    """An ad-hoc way of discovering all SSDP services without explicitly initializing an :class:`~ssdpy.SSDPClient`.

    Returns:
        A list of all discovered SSDP services, each service in a dictionary.
    """
    return SSDPClient().m_search()
