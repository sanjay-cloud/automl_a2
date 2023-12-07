############################################################################
# You do not have to change this file. Do make sure you understand what is
# happening. 
############################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from maml import MAML
from dataloaders import train_val_test_loaders, extract_task

##############################################################################################################
# Parsing command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--num_ways", default=5, type=int, help="Number of classes per task")
argparser.add_argument("--num_support_shots", default=1, type=int, help="Number of examples/shots per class in support sets")
argparser.add_argument("--num_query_shots", default=15, type=int, help="Number of examples/shots per class in query sets")
argparser.add_argument("--meta_batch_size", default=1, type=int, help="Number of tasks in a meta-batch")
argparser.add_argument("--val_interval", default=500, type=int, help="After how many meta-updates we perform meta-validation")
argparser.add_argument("--num_eval_tasks", default=1000, type=int, help="Number of tasks used for validation/testing")
argparser.add_argument("--num_train_episodes", default=40000, type=int, help="Number of meta-updates to make during training")
argparser.add_argument("--lr", default=1e-3, type=float, help="Meta-learning rate (used on query set - potentially acoss tasks)")
argparser.add_argument("--inner_lr", default=0.4, type=float, help="Inner learning rate for MAML (used on support set)")
argparser.add_argument("--second_order", default=False, action="store_true", help="Whether to use second-order gradients")
argparser.add_argument("--dataset", default="omniglot", type=str, help="dataset to use")                                   # DO NOT CHANGE THIS FROM DEFAULT (omniglot)
argparser.add_argument("--T", default=1, type=int, help="Number of inner gradient update steps (inner = on the support set)")
argparser.add_argument("--img_size", type=int, default=28, help="Image size")
argparser.add_argument("--rgb", action="store_true", default=False, help="Use RGB image instead of grayscale") 
argparser.add_argument("--dev", default=None, help="GPU ID to use")
argparser.add_argument("--seed", default=0, type=int, help="seed to use")
args = argparser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
##############################################################################################################

##############################################################################################################
# Create result storage location
RDIR = "./results/"
# check if results directory exists, if not, create it
if not os.path.isdir(RDIR):
    os.mkdir(RDIR)
# the path to where we save the results. we take the first letter of every _ argument block to determine this path
# e.g., num_ways is abbreviated to nw. num_support_shots is abbreviated to nss, and shuffle to s
RDIR = f"./results/" + '-'.join([str(''.join([x[0] for x in item.split('_')])) + str(value) for item,value in vars(args).items()]) + '/'
if not os.path.isdir(RDIR):
    os.mkdir(RDIR)
print(f"Storing results in {RDIR}")
##############################################################################################################

def put_on_device(dev, tensors):
    """
    Puts the given tensors on the given device.

    :param dev (str): the device identifier ("cpu"/"cuda:<GPU-ID>") 
    :param tensors [list]: a list of tensors that we want to put on the device

    :return (list): list of tensors placed on the device
    """
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors

def validate_performance(model, val_loader):
    """
    Function that evaluates the performance of the model on tasks from the given dataloader (either val/test)

    :param model (torch.nn.Module): the model we evaluate
    :param val_loader (dataloader): the validation dataloader 

    :return:
      - vloss (list): list of validation losses on the evaluation tasks
      - vacc (list): list of validation accuracy scores on the evaluation tasks
    """
    vloss, vacc = [], []
    for vid, batch in tqdm(enumerate(val_loader, start=1)):
        
        support_inputs, support_targets, query_inputs, query_targets = extract_task(batch)
        support_inputs, support_targets, query_inputs, query_targets = put_on_device(args.dev, [support_inputs, support_targets, query_inputs, query_targets])
        # compute model predictions on the query set, conditioned on the support set
        preds, loss = model.apply(support_inputs, support_targets, query_inputs, query_targets)
        accuracy = (torch.sum(torch.argmax(preds, dim=1) == query_targets).item()/query_targets.size(0))
        # log scores
        vloss.append(loss.item())
        vacc.append(accuracy)

        if vid == args.num_eval_tasks:
            break

    return vloss, vacc


# Create the data loaders
dataloader_config = {
    "num_ways": args.num_ways,
    "num_support_shots": args.num_support_shots,
    "num_query_shots": args.num_query_shots,
    "ds": args.dataset,
    "img_size": args.img_size,
    "rgb": args.rgb, 
}
train_loader, val_loader, test_loader = train_val_test_loaders(**dataloader_config)

print('cuda available: ', torch.cuda.is_available())
if args.dev is None:
    args.dev = "cpu"
else:
    #args.dev = "cuda:0"
    try:
        torch.cuda.set_device(args.dev)
    except:
        print("Could not connect to GPU 0")
        import sys; sys.exit()

# Define the model that we use
constr = MAML
print(f'Running MAML with {"second-order" if args.second_order else "first-order"} gradients')
model = constr(
            num_ways=args.num_ways,
            T=args.T, 
            input_size=args.img_size**2, 
            second_order=args.second_order,
            rgb=args.rgb, 
            img_size=args.img_size
        ) # images are of size 28x28
model = model.to(args.dev)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr/args.meta_batch_size)


losses = []
accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = -float("inf")
best_parameters = [p.clone().detach() for p in model.parameters()]
force_validation = False
# Main loop
for bid, batch in enumerate(train_loader, start=1):
    # support_inputs shape: (num_support_examples, num channels, img width, img height)
    # support targets shape: (num_support_labels) 

    # query_inputs shape: (num_query_inputs, num channels, img width, img height)
    # query_targets shape: (num_query_labels)
    support_inputs, support_targets, query_inputs, query_targets = extract_task(batch) # extract a single task from the loader
    support_inputs, support_targets, query_inputs, query_targets = put_on_device(args.dev, [support_inputs, support_targets, query_inputs, query_targets])
    
    # compute model predictions on the query set, conditioned on the support set
    preds, loss = model.apply(support_inputs, support_targets, query_inputs, query_targets, training=True)

    if bid % args.meta_batch_size == 0:
        # update the parameters of the model using Adam
        opt.step()
        # zero the gradient buffers for next usage
        opt.zero_grad()

    # log the observed loss and accuracy score
    losses.append(loss.item())
    accuracy = (torch.sum(torch.argmax(preds, dim=1) == query_targets).item()/query_targets.size(0))
    accuracies.append(accuracy)

    if (bid / args.meta_batch_size) >= args.num_train_episodes:
        force_validation = True

    # Meta-validation
    if (bid / args.meta_batch_size) % args.val_interval == 0 or force_validation:
        model.eval()
        vloss, vacc = validate_performance(model, val_loader)
        val_losses.append(vloss)
        val_accuracies.append(vacc)

        avg_val_acc = np.mean(vacc)
        print("Mean validation accuracy:", avg_val_acc, "mean training accuracy:", np.mean(accuracies))
        # if we exceed the incumbent best performance, store a copy of the weights so that we can use them for testing
        if avg_val_acc > best_val_acc:
            best_parameters = [p.clone().detach() for p in model.parameters()]
            best_val_acc = avg_val_acc
        model.train()

    if force_validation:
        break

        
# Load the best found parameters from validation
for current_param, best_param in zip(model.parameters(), best_parameters):
    current_param.data = best_param.data

model.eval()
test_losses, test_accuracies = validate_performance(model, test_loader)


np.save(RDIR + "train-loss.npy", losses)
np.save(RDIR + "train-accuracy.npy", accuracies)
np.save(RDIR + "val-loss.npy", val_losses)
np.save(RDIR + "val-accuracy.npy", val_accuracies)
np.save(RDIR + "test-loss.npy", test_losses)
np.save(RDIR + "test-accuracy.npy", test_accuracies)

# You can also add code in case you want to store the best model parameters