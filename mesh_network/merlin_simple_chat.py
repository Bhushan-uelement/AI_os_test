import socket
import threading
import sys

# Configuration
PORT = 5555 

def receive_messages(sock):
    """Continuously listens for incoming data."""
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            # addr[0] is the sender's IP address
            print(f"\n[Drone {addr[0]}]: {data.decode('utf-8')}")
            print("You: ", end="", flush=True)
        except Exception as e:
            print(f"\n[Error] Receiver stopped: {e}")
            break

def main():
    print("--- MerlinOS Simple Mesh Chat ---")
    
    target_ip = input("Enter Target IP (Pi or QEMU): ").strip()
    
    # Create a UDP Socket
    # AF_INET = IPv4, SOCK_DGRAM = UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Bind to '0.0.0.0' so it listens on all interfaces (wlan0, bat0, etc.)
    try:
        sock.bind(("0.0.0.0", PORT))
    except Exception as e:
        print(f"Could not bind to port {PORT}: {e}")
        return

    # Start the background listener thread
    listener = threading.Thread(target=receive_messages, args=(sock,), daemon=True)
    listener.start()

    print(f"Ready! Sending to {target_ip} on port {PORT}")
    print("Type your message and press Enter. (Ctrl+C to exit)\n")

    try:
        while True:
            msg = input("You: ")
            if msg.strip():
                sock.sendto(msg.encode('utf-8'), (target_ip, PORT))
    except KeyboardInterrupt:
        print("\nExiting MerlinChat...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
