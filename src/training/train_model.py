import pickle
import os
import argparse
import json
import datetime
import jax

from src.data import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100, get_dataloader, n_classes
from src.models import MLP, LeNet
from src.training.maximum_a_posteriori import maximum_a_posteriori



parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], default="MNIST")
parser.add_argument("--data_path", type=str, default="/dtu/p1/hroy/data/", help="root of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--train_val_ratio", type=float, default=0.9)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=2, type=int, help="Number of layers in the MLP.")
# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "rmsprop"], default="adam")
parser.add_argument("--prior_precision", type=float, default=1e-5)
parser.add_argument("--likelihood_precision", type=float, default=1.)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification"], default="classification")
# not affecting anything
parser.add_argument("--test_every_n_epoch", type=int, default=1e10)
# storage
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--model_save_path", type=str, default="./checkpoints/", help="Root where to save models")



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()
    args_dict = vars(args)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    ###############
    ### dataset ###
    n_samples_per_class = None if args.n_samples is None else int(args.n_samples/n_classes(args.dataset))
    cls=list(range(n_classes(args.dataset)))
    if args.dataset == "MNIST":
        dataset = MNIST(path_root=args.data_path, train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=args.seed)
    elif args.dataset == "FMNIST":
        dataset = FashionMNIST(path_root=args.data_path, train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=args.seed)
    elif args.dataset == "CIFAR-10":
        dataset = CIFAR10(path_root=args.data_path, train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=args.seed)
    elif args.dataset == "CIFAR-100":
        dataset = CIFAR100(path_root=args.data_path, train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=args.seed)
    elif args.dataset == "SVHN":
        dataset = SVHN(path_root=args.data_path, train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=args.seed)
    dataset_size = len(dataset)
    train_loader, valid_loader = get_dataloader(
        dataset, 
        train_size=int(args.train_val_ratio * dataset_size), 
        batch_size=args.batch_size, 
        num_workers=5, 
        pin_memory=False, 
        drop_last=False,
        shuffle=True, 
        seed=args.seed)
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")

    #############
    ### model ###
    output_dim = n_classes(args.dataset)
    if args.model == "MLP":
        model = MLP(output_dim=output_dim, num_layers=args.mlp_num_layers, hidden_dim=args.mlp_hidden_dim, activation=args.activation_fun)
    elif args.model == "LeNet":
        model = LeNet(output_dim=output_dim, activation=args.activation_fun)
    else:
        raise ValueError(f"Model {args.model} is not implemented")

    ################
    ### training ###
    params, stats_dict = maximum_a_posteriori(
            model, train_loader, valid_loader, jax.random.PRNGKey(args.seed), args_dict
        )
    model_dict = {
        "params": params,
        "prior_precision": args.prior_precision,
        "likelihood_precision": args.likelihood_precision,
        "model": args.model
    }

    ####################################
    ### save params and dictionaries ###

    # first folder is dataset
    save_folder = f"{args.model_save_path}/{args.dataset}"
    if args.n_samples is not None:
        save_folder += f"_samples{args.n_samples}"
    # second folder is model
    if args.model == "MLP":
        save_folder += f"/MLP_depth{args.mlp_num_layers}_hidden{args.mlp_hidden_dim}"
    else:
        save_folder += f"/{args.model}"
    os.makedirs(save_folder, exist_ok=True)
    
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.seed}"
    else:
        save_name = f"started_{now_string}"

    print(f"Saving to {save_folder}/{save_name}")

    pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
    pickle.dump(stats_dict, open(f"{save_folder}/{save_name}_stats.pickle", "wb"))
    with open(f"{save_folder}/{save_name}_args.json", "w") as f:
        json.dump(args_dict, f)