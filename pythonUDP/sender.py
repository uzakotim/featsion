from time import sleep
import sys
import termios
import socket

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

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver_address = ('127.0.0.1', 8090)

while (True):
    msg = getchar()
    print(msg)
    data = f'{msg}'.encode()
    sock.sendto(data,receiver_address)
    sleep(0.1)
    if msg == 'q':
        break

sock.close()