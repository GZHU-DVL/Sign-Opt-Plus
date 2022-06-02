import torch
import numpy as np
import torchvision.transforms.functional as F


class PytorchModel(object):
    def __init__(self,model, bounds, num_classes, dataset, stand):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
        self.num_queries = 0
        self.dataset = dataset
        self.stand = stand

    def standard(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_tmp = image.clone()
        if (self.dataset == 'ImageNet'):
            res = torch.stack([F.normalize(image_tmp[0], mean, std)])
            return res

    def predict_label(self, image, batch=False):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        if (self.dataset == 'ImageNet' and self.stand):
            image = self.standard(image)

        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        with torch.no_grad():
            output = self.model(image)
            self.num_queries += image.size(0)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        _, predict = torch.max(output.data, 1)
        if batch:
            return predict
        else:
            return predict[0]

