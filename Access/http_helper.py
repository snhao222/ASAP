# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

def parse_headers(response):
    """Receives an HTTP response/request bytes object and parses the HTTP headers.

    Args:
        response (bytes object): HTTP response/request bytes object.
    
    Returns:
        a dict of all headers.
    """
    valid_headers = (
        b"NOTIFY * HTTP/1.1\r\n",
        b"M-SEARCH * HTTP/1.1\r\n",
        b"HTTP/1.1 200 OK\r\n",
    )
    if not any([response.startswith(x) for x in valid_headers]):
        raise ValueError(
            "Invalid header: Should start with one of: {}".format(valid_headers)
        )

    lines = response.split(b"\r\n")
    headers = {}
    # Skip the first line since it's just the HTTP return code
    for line in lines[1:]:
        if not line:
            break  # Headers and content are separated by a blank line
        if b":" not in line:
            raise ValueError("Invalid header: {}".format(line))
        header_name, header_value = line.split(b":", 1)
        headers[header_name.decode("utf-8").lower().strip()] = header_value.decode(
            "utf-8"
        ).strip()
    return headers