import pickle
import json
import jax
from jax import flatten_util
from models import MLP, LeNet

def load_model(
        dataset = "MNIST",
        seed = 0,
        run_name = "good",
        model_name = "LeNet",
        model_save_path = "../models/"
    ):

    args_file_path = f"{model_save_path}/{dataset}/{model_name}/{run_name}_seed{seed}_args.json"
    args_dict = json.load(open(args_file_path, 'r'))
    assert dataset == args_dict["dataset"]

    if args_dict["dataset"] in ["Sinusoidal", "UCI"]:
        output_dim = 1 
    elif args_dict["dataset"] == "CelebA":
        output_dim = 40
    else:
        output_dim = 10
    if args_dict["model"] == "MLP":
        model = MLP(
            output_dim=output_dim, 
            num_layers=args_dict["mlp_num_layers"], 
            hidden_dim=args_dict["mlp_hidden_dim"], 
            activation=args_dict["activation_fun"]
        )
    elif args_dict["model"] == "LeNet":
        model = LeNet(
            output_dim=output_dim,
            activation=args_dict["activation_fun"]
        )

    params_file_path = f"{model_save_path}/{dataset}/{model_name}/{run_name}_seed{seed}_params.pickle"
    params_dict = pickle.load(open(params_file_path, 'rb'))
    params = params_dict['params']

    
    P = flatten_util.ravel_pytree(params)[0].shape[0]
    print(f"Loaded {args_dict['model']} with {P} parameters")

    return model, params