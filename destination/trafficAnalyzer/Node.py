from collections import defaultdict

from . import Stats
from . import Constants


class Nodes(object):
    def __init__(self):
        self.nodes = {}

    def __getitem__(self, nodeId):
        if nodeId not in self.nodes:
            self.nodes[nodeId] = NodeStats(nodeId)
    
        return self.nodes[nodeId]

    def __contains__(self, nodeId):
        if nodeId in self.nodes:
            return True
        return False


class NodeStats(object):
    def __init__(self, node_id, base_ts=0, devices=None):
        self.nodeId = node_id
        self.desc = ""
        self.baseTS = base_ts
        self.devices = devices
        self.stats = Stats.Stats(self)
        self.extractLayers()

    def processPacket(self, packet):
        #print(dir(packet.eth))
        if packet.eth.src == self.nodeId.mac:
            self.proc_pckt(packet, Constants.Direction.RCV, Constants.Direction.SND)
        else:
            self.proc_pckt(packet, Constants.Direction.SND, Constants.Direction.RCV)

    def proc_pckt(self, packet, dir1, dir2):
        addr = NodeId()
        addr.extractFromPacket(packet, dir1, self.devices)
        packet.addr = addr
        for layer in packet.layers:
            if layer.layer_name not in self.layersToProcess:
                continue
            stats = self.stats.getStats(layer.layer_name, dir2)
            stats.processLayer(packet, layer)

    def extractLayers(self):
        layers = defaultdict(int)
        layers['eth'] += 1
    
        self.layersToProcess = ['eth'] #dict.keys(layers)


class NodeId(object):
    def __init__(self, mac=None, ip=None, time=0):
        self.mac = mac
        self.ip = ip
        self.deviceName = None
        self.ipHistory = []
        if ip is not None:
            self.ipHistory.append((ip, time))
    
    def setMacIp(self, mac, ip, time=0):
        self.mac = mac
        self.ip = ip
        self.ipHistory.append((ip, time))

    def addIP(self, ip, time):
        self.ip = ip
        self.ipHistory.append((ip, time))

    def extractFromPacket(self, packet, direction, devices):
        if direction == Constants.Direction.SND:
            self.mac = packet.eth.src
            self.deviceName = devices.getDeviceName(self.mac)
            try:
                self.ip = packet.ip.src
            except AttributeError:
                pass
        else:
            self.mac = packet.eth.dst
            self.deviceName = devices.getDeviceName(self.mac)
            try:
                self.ip = packet.ip.dst
            except AttributeError:
                pass

    def getAddr(self):
        if self.deviceName is not None:
            return self.deviceName
        elif self.ip is not None:
            return self.ip
        return self.mac

    def __str__(self):
        return "mac: {} ip: {} history ip: {}".format(self.mac, self.ip, self.ipHistory)

