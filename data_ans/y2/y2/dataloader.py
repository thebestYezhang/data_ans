import torch
import torchvision
import folders

class DataLoader(object):
    """Dataset class for Mnist databases"""

    def __init__(self, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain


        # Train
        if istrain:
            self.data = folders.Folder(train=True)
        # Test
        else:
            self.data = folders.Folder(train=False)




    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader