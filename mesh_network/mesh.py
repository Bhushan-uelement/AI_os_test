import os
import socket
import threading
import time
import netifaces

# --- CONFIGURATION ---
INTERFACE = "wlp0s20f3"
MESH_ID = "private_spectrum"
FREQUENCY = "5180"  # Channel 36 (5GHz)
PORT = 5555
BROADCAST_PORT = 5556

class MeshChat:
    def __init__(self):
        self.peers = {} # Format: {ip: hostname}
        self.my_ip = ""
        
    def setup_hardware(self):
        print(f"[*] Configuring {INTERFACE} for Private Mesh...")
        commands = [
            f"ip link set {INTERFACE} down",
            f"iw dev {INTERFACE} set type mesh",
            f"ip link set {INTERFACE} up",
            f"iw dev {INTERFACE} mesh join {MESH_ID} freq {FREQUENCY}",
            # Assign a static IP based on MAC to avoid collisions (Simple hack)
            f"ip addr add 192.168.10.{self.get_ip_suffix()}/24 dev {INTERFACE}"
        ]
        for cmd in commands:
            os.system(cmd)
        time.sleep(2)
        self.my_ip = self.get_local_ip()
        print(f"[+] Hardware Ready. IP: {self.my_ip}")

    def get_ip_suffix(self):
        # Generates a unique IP suffix from the MAC address
        mac = netifaces.ifaddresses(INTERFACE)[netifaces.AF_LINK][0]['addr']
        return int(mac.split(':')[-1], 16)

    def get_local_ip(self):
        return netifaces.ifaddresses(INTERFACE)[netifaces.AF_INET][0]['addr']

    # --- DISCOVERY LOGIC ---
    def discovery_broadcaster(self):
        """Tells everyone: 'I am here!'"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            msg = f"DISCOVER:{socket.gethostname()}"
            sock.sendto(msg.encode(), ('192.168.10.255', BROADCAST_PORT))
            time.sleep(5)

    def discovery_listener(self):
        """Listens for other devices 'shouting'"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', BROADCAST_PORT))
        while True:
            data, addr = sock.recvfrom(1024)
            ip = addr[0]
            if ip != self.my_ip:
                name = data.decode().split(':')[-1]
                if ip not in self.peers:
                    print(f"\n[!] New Peer Discovered: {name} ({ip})")
                    self.peers[ip] = name

    # --- CHAT LOGIC ---
    def receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', PORT))
        while True:
            data, addr = sock.recvfrom(1024)
            print(f"\n[{self.peers.get(addr[0], addr[0])}]: {data.decode()}")
            print("> ", end="", flush=True)

    def sender(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            msg = input("> ")
            if msg.lower() == 'list':
                print(f"Connected Peers: {self.peers}")
                continue
            
            # Send to all discovered peers
            for ip in self.peers:
                sock.sendto(msg.encode(), (ip, PORT))

    def start(self):
        self.setup_hardware()
        
        # Start Threads
        threading.Thread(target=self.discovery_broadcaster, daemon=True).start()
        threading.Thread(target=self.discovery_listener, daemon=True).start()
        threading.Thread(target=self.receiver, daemon=True).start()
        
        print("--- Mesh Chat Active ---")
        print("Type messages to send to all peers. Type 'list' to see peers.")
        self.sender()

if __name__ == "__main__":
    chat = MeshChat()
    chat.start()
