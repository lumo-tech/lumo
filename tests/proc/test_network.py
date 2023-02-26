from http.server import HTTPServer, BaseHTTPRequestHandler

from lumo.proc.network import find_free_network_port


def test_port():
    res = []
    try:
        port = find_free_network_port()
        server_address = ('', port)
        httpd = HTTPServer(server_address, BaseHTTPRequestHandler)
        res.append(httpd)
    except OSError:
        raise AssertionError('cannot use')

    for r in res:
        r.server_close()
