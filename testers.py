import torch
import preprocess
from preprocess import device
from preprocess import train_data, val_data


# function for getting batches of data
def get_tester_batch(split, context_window_size, device, batch_size=32):
    """
    generate a small batch of data of inputs x and targets y

    Args:
        split: 'train' or 'val'
        device: 'cpu' or 'cuda' (should be 'cuda' if available)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_window_size, (batch_size,))
    x = torch.stack([data[i:i+context_window_size] for i in ix])
    x = x.to(device)
    return x



# collect word length
@torch.no_grad()
def word_length(tensor, space):
    dist = []
    for row in tensor:
        row = torch.cat([torch.tensor([space[0]], device=device), row, torch.tensor([space[0]], device=device)])
        indices = (row == space[0]) + (row == space[1])
        indices = indices.to(torch.int).nonzero(as_tuple=True)[0]
        if torch.max(torch.diff(indices)) > 30:
            print(preprocess.decode(row.tolist()))
        dist.append(torch.bincount(torch.diff(indices), minlength=250))
        
    return torch.stack(dist)


# number of capital letters in 'weird' spots
@torch.no_grad()
def capital_counts(tensor, capitals, new_line):
    '''
    tensor: 2D
    space: integer
    capitals: integer list
    '''

    numbers = []
    for row in tensor:
        index = (torch.cat([torch.tensor([new_line], device=device), row[:-1]]) == new_line)
        ls1 = [row[index] == v for v in capitals]
        ls2 = [row.masked_select(index) == v for v in capitals]
        numbers.append(torch.sum(index) - torch.sum(torch.stack(ls1)) + torch.sum(torch.stack(ls2)))

    return torch.tensor(numbers)


