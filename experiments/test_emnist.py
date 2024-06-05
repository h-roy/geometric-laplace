import torchvision
from torchvision.datasets.utils import download_url
from torchvision import transforms

import os

raw_folder = '.data/EMNIST/raw'

url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

version_numbers = list(map(int, torchvision.__version__.split('+')[0].split('.')))
if version_numbers[0] == 0 and version_numbers[1] < 10:
    filename = "emnist.zip"
else:
    filename = None

os.makedirs(raw_folder, exist_ok=True)

# download files
print('Downloading zip archive')
download_url(url, root=raw_folder, filename=filename, md5=md5)

# Update the path to where you've manually placed the EMNIST dataset
root_dir = ".data/EMNIST/raw"  # Change this to the actual path

trainset = torchvision.datasets.EMNIST(root=root_dir,
                                   split="letters",
                                   train=True,
                                   download=False,  # Set to False since you already downloaded it 
                                   transform=transforms.ToTensor())
