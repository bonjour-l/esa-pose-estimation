import socket
import datetime
from time import sleep


def get_currenttime():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return now_time
class tcp_send():
    def __init__(self,host,port):
        self.proname='esa'
        self.time=get_currenttime()
        self.host=host
        self.port=port
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    def create_socket(self,classname='cat'):

        try:
            self.sock.connect((self.host,self.port))
        except socket.error as e:
            print(e)
            return False

        self.send(self.time,type='log',classname=classname)
        self.send(self.time,type='load',classname=classname)
        return True

    def send(self,str,type='log',classname='cat'):

        data='\\runlog\r\nproname:'+self.proname+'\r\n'
        data=data+('ltype:'+type+'\r\n')
        data=data+('classname:'+classname+'\r\n')
        data=data+('data:'+str)
        data+='\0'
        self.sock.sendall(data.encode())

    def close(self):
        self.sock.close()


if __name__=='__main__':
    print(get_currenttime())
    sock=tcp_send('182.92.72.74',6000)
    sock.create_socket()
    for i in range(10):
        sock.send("123")
        sleep(0.5)
    sock.close()
