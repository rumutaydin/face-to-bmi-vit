import os
import urllib.request
from PIL import Image
#from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms

import subprocess
import mimetypes

def test_and_show(img_dir, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ToTensor()(image)
    image_vit = vit_transforms(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    layers = []

    layers.append(torch.nn.Linear(768, 285))
    layers.append(torch.nn.SiLU())
    layers.append(torch.nn.Dropout(0.5))
    layers.append(torch.nn.Linear(285, 187))
    layers.append(torch.nn.SiLU())
    layers.append(torch.nn.Dropout(0.5))
    layers.append(torch.nn.Linear(187, 133))
    layers.append(torch.nn.SiLU())
    layers.append(torch.nn.Linear(133, 1))

    model_head = torch.nn.Sequential(*layers).to(device)
    model = get_model()
    model.heads = model_head

    model = model.to(device)

    if os.path.exists(weight_dir):
        file_size = os.path.getsize(weight_dir)
        print(f"Downloaded file size: {file_size} bytes")
    try:
        model.load_state_dict(torch.load(weight_dir, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    # plot
    # plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    # plt.axis("off")
    # plt.title(f"Predicted BMI: {pred.item():>5f}")
    # plt.show()

    return pred.item()



def verify_downloaded_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return False
    
    file_size = os.path.getsize(file_path)
    if file_size < 100 * 1024 * 1024:  # Check if file is smaller than 100 MB
        print(f"File {file_path} is unexpectedly small (size: {file_size} bytes).")
        return False

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type not in ['application/x-tar', 'application/octet-stream']:
        print(f"File {file_path} is not a valid model checkpoint (detected mime type: {mime_type}).")
        return False
    
    return True


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_from_gcs_with_subprocess(bucket_name, source_blob_name, destination_file_name):
    try:
        command = ["gsutil", "cp", f"gs://{bucket_name}/{source_blob_name}", destination_file_name]
        subprocess.run(command, check=True)
        print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download file from GCS: {e}")

if __name__ == "__main__":
    #image_url = "https://media.licdn.com/dms/image/C5603AQHdJoPNANFgGw/profile-displayphoto-shrink_800_800/0/1516822371332?e=1726704000&v=beta&t=f_HtGhME8limB7AyjfPhfRylF9Y9Tso4e0387-1IJN4"
    image_path = '../data/test_pic.jpg'
    
    if not os.path.exists("../data"):
        os.makedirs("../data")
    #urllib.request.urlretrieve(image_url, image_path)

    if not os.path.exists("../inf_weights"):
        os.makedirs("../inf_weights")

    weight_dir = "../inf_weights/best.pt"
    if not os.path.exists(weight_dir):
        bucket_name = "mimic-public"
        source_blob_name = "bmi_prediction/saved_models/lr0.00289-bs128-hidden3-oadamw-nswish.pt"
        download_from_gcs_with_subprocess(bucket_name, source_blob_name, weight_dir)


    pred = test_and_show(image_path, weight_dir)
    if pred is not None:
        print(f'Predicted BMI: {pred}')
