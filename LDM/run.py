import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from src.ldm import LDM
from src import utils

# Global control for device
CUDA = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--dataset', type=str, required=True, help='Path of the dataset'
    )
    parser.add_argument(
        '--emb_path', type=str, required=True, help='Path of the embedding file'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=200, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=10, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )
    parser.add_argument(
        '--visualize', type=bool, default=1, required=False, help='Verbose'
    )

    return parser.parse_args()


def process(args):

    dataset_path = args.dataset
    emb_path = args.emb_path
    dim = args.dim
    epoch_num = args.epoch_num
    steps_per_epoch = args.spe
    batch_size = args.batch_size
    lr = args.lr

    seed = args.seed
    verbose = args.verbose

    edges = utils.read_emb(dataset_path)
    edges = torch.as_tensor(edges, dtype=torch.int, device=torch.device("cpu")).T

    # Run the model
    ldm = LDM(
        edges=edges, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch,
        device=torch.device(device), verbose=verbose, seed=seed
    )
    # Learn the embeddings
    ldm.learn()
    # Save the model
    ldm.save_embs(path=emb_path)

    if args.visualize:
        # Visualize the embeddings
        utils.visualize(ldm.get_embs().detach().cpu().numpy())
        

def process_debug(args):
    dataset_path = args[0]
    emb_path = args[1]
    dim = args[2]
    epoch_num = args[3]
    steps_per_epoch = args[4]
    batch_size = args[5]
    lr = args[6]

    seed = args[7]
    verbose = args[8]
    
    edges = utils.read_emb(dataset_path)
    edges = torch.as_tensor(edges, dtype=torch.int, device=torch.device("cpu")).T

    # Run the model
    ldm = LDM(
        edges=edges, dim=dim, lr=lr, epoch_num=epoch_num, batch_size=batch_size, spe=steps_per_epoch,
        device=torch.device(device), verbose=verbose, seed=seed
    )
    # Learn the embeddings
    ldm.learn()
    # Save the model
    ldm.save_embs(path=emb_path)

    if args.visualize:
        # Visualize the embeddings
        utils.visualize(ldm.get_embs().detach().cpu().numpy())
    


if __name__ == "__main__":
    #args = parse_arguments()
    #process(args)
    
    args = ['./Data/paper2paper_edgelist', './ldm_paper2papertest.emb', 2, 200, 10, 50, 0.1, 19, 1, 1]

    process_debug(args)