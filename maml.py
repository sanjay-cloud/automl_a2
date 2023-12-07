import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Conv4
from torchmeta.utils.gradient_based import gradient_update_parameters


class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 


    # controller input = image + label_previous
    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Pefrosmt the inner-level learning procedure of MAML: adapt to the given task 
        using the support set. It returns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters 

        :param x_supp (torch.Tensor): the support input iamges of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """
        
        # TODO: implement this function
        for i in range(self.num_updates):
            supp_pred = self.network(x_supp)
            supp_loss = self.inner_loss(supp_pred, y_supp)            
            params = [] 
            grads = torch.autograd.grad(supp_loss, self.network.parameters(), create_graph = self.second_order)
            for param, grad in zip(self.network.parameters(), grads):
                params.append(param - self.inner_lr * grad)

        query_pred = self.network(x_query, weights = params)  
        query_loss = self.inner_loss(query_pred, y_query)  
        # Note: to make predictions and to allow for second-order gradients to flow if we want,
        # we use a custom forward function for our network. You can make predictions using
        # preds = self.network(input_data, weights=<the weights you want to use>)

        if training:
            query_loss.backward() # do not remove this if statement, otherwise it won't train
            
        return query_pred, query_loss