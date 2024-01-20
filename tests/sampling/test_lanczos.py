import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from src.data import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100, n_classes
from src.models.utils import load_model
from src.helper import get_gvp_fun, get_ggn_vector_product
from src.sampling import lanczos_tridiag
from src.losses import cross_entropy_loss

dataset_name = "CIFAR-10"

model_name = "LeNet"
run_name = "example"
model_save_path = "./checkpoints/"
dataset_save_path = "/work3/hroy/data/"

lanczos_iter = 100




###############
### dataset ###
cls=list(range(n_classes(dataset_name)))
if dataset_name == "MNIST":
    dataset = MNIST(path_root=dataset_save_path, train=True, n_samples_per_class=None, download=False, cls=cls, seed=0)
elif dataset_name == "FMNIST":
    dataset = FashionMNIST(path_root=dataset_save_path, train=True, n_samples_per_class=None, download=False, cls=cls, seed=0)
elif dataset_name == "CIFAR-10":
    dataset = CIFAR10(path_root=dataset_save_path, train=True, n_samples_per_class=None, download=False, cls=cls, seed=0)
elif dataset_name == "CIFAR-100":
    dataset = CIFAR100(path_root=dataset_save_path, train=True, n_samples_per_class=None, download=False, cls=cls, seed=0)
elif dataset_name == "SVHN":
    dataset = SVHN(path_root=dataset_save_path, train=True, n_samples_per_class=None, download=False, cls=cls, seed=0)
data_array = jnp.array([data[0] for data in dataset])
print(f"Loaded {dataset_name} with shape {data_array.shape}")

#############
### model ###
model, params = load_model(
    dataset = dataset_name,
    seed = 0,
    run_name = run_name,
    model_name = model_name,
    model_save_path = model_save_path
)
vectorize_fun = lambda x : jax.flatten_util.ravel_pytree(x)[0]
devectorize_fun = jax.flatten_util.ravel_pytree(params)[1]
num_params = vectorize_fun(params).shape[0]




##########################
### GGN vector product ###
# x_train = data_array[:100]
# y_train = jnp.array([data[1] for data in dataset])[:100]
# ggn_vector_product_1 = get_gvp_fun(
#         model.apply,
#         cross_entropy_loss,
#         params,
#         x_train,
#         y_train
# )
# random_vector = jax.random.normal(jax.random.PRNGKey(0), shape=(num_params, ))

# start = time.time()
# ggn_vector_product_1(random_vector)
# print(f"\nOne GGN vector product (with {len(data_array)} datapoints) took {time.time()-start:.5f} seconds")

# print("...Let's try again to exploit compilation...")
# random_vector = jax.random.normal(jax.random.PRNGKey(1), shape=(num_params, ))
# start = time.time()
# ggn_vector_product_1(random_vector)
# print(f"One GGN vector product (with {len(data_array)} datapoints) took {time.time()-start:.5f} seconds")



###############
### Lanczos ###
# start = time.time()
# eigenval_1, eigenvec = lanczos_tridiag(ggn_vector_product_1, random_vector, lanczos_iter)
# print(f"{lanczos_iter} iterations of Lanczos took {time.time()-start:.3f} seconds")

#### Other GGN vector product
ggn_vector_product_2 = get_ggn_vector_product(
                                                params,
                                                model,
                                                data_array = data_array,
                                                likelihood_type = "classification"
                                        )

random_vector = jax.random.normal(jax.random.PRNGKey(0), shape=(num_params, ))

start = time.time()
ggn_vector_product_2(random_vector)
print(f"\nOne GGN vector product (with {len(data_array)} datapoints) took {time.time()-start:.5f} seconds")

print("...Let's try again to exploit compilation...")
random_vector = jax.random.normal(jax.random.PRNGKey(1), shape=(num_params, ))
start = time.time()
ggn_vector_product_2(random_vector)
print(f"One GGN vector product (with {len(data_array)} datapoints) took {time.time()-start:.5f} seconds")



###############
### Lanczos ###
start = time.time()
eigenval_2, eigenvec = lanczos_tridiag(ggn_vector_product_2, random_vector, lanczos_iter)
print(f"{lanczos_iter} iterations of Lanczos took {time.time()-start:.3f} seconds")

# plt.plot(eigenval_1)
plt.plot(eigenval_2)
plt.yscale("log")
plt.savefig(f"eigenvalues_{dataset_name}_{model_name}")