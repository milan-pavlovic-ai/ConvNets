
import torch


class DeviceMngr():
    """
    Class for setting device and moving data to specific device
    """

    def __init__(self):
        super().__init__()
        self.device = self.get_default_device()

    def get_default_device(self):
        """
        Returns first GPU device if it's available, otherwise returns CPU device
        """
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('You are using device:', device_str)
        return torch.device(device_str)

    def set_device(self, device_str):
        """
        Set specific device by given string
        """
        self.device = torch.device(device_str)
        print('You are using device:', device_str)
        return

    def move(self, data):
        """
        Move given data/tensors to default device
        """
        if isinstance(data, (list,tuple)):
            return [self.move(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def move_to(self, data, device_str):
        """
        Move give data/tensors to given device
        """
        device = torch.device(device_str)
        return data.to(device, non_blocking=True)
   

if __name__ == "__main__":
    device = DeviceMngr()
    device.device
    device.set_device('cpu')
    