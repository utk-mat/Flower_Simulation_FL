from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from model import Net, train, test 

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, 
                 trainloader, 
                 valloader,
                 num_classes: int
                ) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)



    def get_parameters(self, config: Dict[str, Scalar]):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        



    def fit(self, parameters, config):

        # copy the parameters sent by the server into the client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.trainloader, optim,epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):

        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}
    


def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(cid: str):

        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,)
    

    return client_fn