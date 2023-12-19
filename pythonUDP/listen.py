from time import sleep
import sys
import termios
import socket

host = "0.0.0.0"  # Listen on all available interfaces
port = 8080       # Choose a suitable port number
def getchar():
    old = termios.tcgetattr(sys.stdin)
    cbreak = old.copy()
    cbreak[3] &= ~(termios.ECHO|termios.ICANON)
    cbreak[6][termios.VMIN] = 1
    cbreak[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin,termios.TCSADRAIN,cbreak)
    char = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin,termios.TCSADRAIN,old)
    return char

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        # Bind the socket to a specific address and port
        udp_socket.bind((host, port))
        print(f"Listening for UDP messages on {host}:{port}")
        while True:
            # Receive data from the socket
            data, addr = udp_socket.recvfrom(1024)
            [command, speed] = data.decode('utf-8').split()
            speed = int(speed)
            print([command, speed])
            sleep(0.01)