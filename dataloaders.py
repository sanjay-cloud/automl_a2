###################################################################################
# DO NOT CHANGE THIS FILE
###################################################################################
import torch


from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from torchmeta.utils.data import BatchMetaDataLoader

def train_val_test_loaders(num_ways, num_support_shots, num_query_shots, ds="omniglot", img_size=28, rgb=False):
    if ds.lower() == "omniglot":
        ds_constr = Omniglot
        transform = Compose([Resize(img_size), ToTensor()])
    else:
        ds_constr = MiniImagenet
        transform = [Resize(img_size), ToTensor()]
        if not rgb:
            transform.append(Grayscale())
        transform = Compose(transform)
    
    print("Using constr:", ds_constr)

    dataset = ds_constr("data",
                   # Number of ways
                   num_classes_per_task=num_ways,
                   # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                   transform=transform,
                   # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                   target_transform=Categorical(num_classes=num_ways),
                   # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                   class_augmentations=None,
                   meta_train=True,
                   download=True)
    dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=num_support_shots, num_test_per_class=num_query_shots)
    dataloader = BatchMetaDataLoader(dataset, batch_size=1, num_workers=0)


    val_dataset = ds_constr("data",
                    # Number of ways
                    num_classes_per_task=num_ways,
                    # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                    transform=transform,
                    # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                    target_transform=Categorical(num_classes=num_ways),
                    # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                    class_augmentations=None,
                    meta_val=True,
                    download=True)
    val_dataset = ClassSplitter(val_dataset, shuffle=True, num_train_per_class=num_support_shots, num_test_per_class=num_query_shots)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=1, num_workers=0)


    test_dataset = ds_constr("data",
                    # Number of ways
                    num_classes_per_task=num_ways,
                    # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                    transform=transform,
                    # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                    target_transform=Categorical(num_classes=num_ways),
                    # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                    class_augmentations=None,
                    meta_test=True,
                    download=True)
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=num_support_shots, num_test_per_class=num_query_shots)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=0)

    return dataloader, val_dataloader, test_dataloader

def extract_task(batch, shuffle=False):
    """
    Extract a task (support data, query data) from a batch from the dataloader
    DO NOT CHANGE THIS FUNCTION.
    """
    support_inputs, support_targets = batch["train"]
    query_inputs, query_targets = batch["test"]

    if shuffle:
        sindices = torch.randperm(support_inputs.size(1))
        qindices = torch.randperm(query_inputs.size(1))

        support_inputs, support_targets = support_inputs[:,sindices,:,:], support_targets[:,sindices]
        query_inputs, query_targets = query_inputs[:,qindices,:,:], query_targets[:,qindices]

    return support_inputs[0], support_targets[0], query_inputs[0], query_targets[0]