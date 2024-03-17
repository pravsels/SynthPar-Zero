import torch
import numpy as np
import dnnlib
import os 
from stylegan_utils.stylegan2 import Generator as StyleGAN2Generator
from stylegan_utils import legacy # pylint: disable=import-error
from PIL import Image
from torch.utils.data import DataLoader
from utils import parse_arguments, get_config
from utils import CapturedException, CustomDataset
from utils import class_to_race_map

class Generator:
    def __init__(self, 
                 pkl, 
                 device='cuda', 
                 imgs_dir='generated_images'):
        
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._dtype = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data = None
        self._G = None
        self.pkl = pkl
        self.imgs_dir = imgs_dir
        self.load_network()

    def load_network(self):
        print(f'Loading "{self.pkl}"... ', end='', flush=True)
        try:
            with dnnlib.util.open_url(self.pkl, verbose=False) as f:
                self._pkl_data = legacy.load_network_pkl(f)
            print('Done.')
        except:
            raise CapturedException(f'Failed to load "{self.pkl}"!')

        try:
            self._G = StyleGAN2Generator(*self._pkl_data['G_ema'].init_args, 
                                         **self._pkl_data['G_ema'].init_kwargs)
            self._G.load_state_dict(self._pkl_data['G_ema'].state_dict())
            self._G.to(self._device)
            self._G.eval()
        except:
            raise CapturedException(f'Failed to initialize generator from "{self.pkl}"!')

    def generate_images(self, 
                        w0_seed=0, 
                        truncation_psi=0.7, 
                        truncation_cutoff=None, 
                        class_index=None):
        
        z = torch.cat([torch.from_numpy(np.random.RandomState(ws).randn(1, 512)).to(self._device, dtype=self._dtype) for ws in w0_seed], dim=0)
        
        label = torch.zeros([w0_seed.shape[0], self._G.c_dim], device=self._device)
        
        if class_index is not None:
            label[:, class_index] = 1
            output_dir = os.path.join(self.imgs_dir, class_to_race_map[class_index])
            os.makedirs(output_dir, 
                        exist_ok=True)
        
        ws = self._G.mapping(z, 
                             label, 
                             truncation_psi=truncation_psi, 
                             truncation_cutoff=truncation_cutoff)
        imgs = self._G.synthesis(ws)
        
        # Permute the dimensions to [batch_size, H, W, C] for each image in the batch
        imgs = imgs.permute(0, 2, 3, 1)
        # Convert the images to uint8 format within the 0-255 range
        imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        pil_imgs = [Image.fromarray(img.cpu().numpy()) for img in imgs]
        
        for index, p_img in zip(w0_seed, pil_imgs):
            p_img.save(f'{output_dir}/{index}.png')


if __name__ == "__main__":
    command_line_args = parse_arguments()
    config = get_config(command_line_args.config)

    batch_size = 8 
    num_workers = 1 

    class_index = config.class_index
    no_of_identities_per_class = config.no_of_identities_per_class

    dataset = CustomDataset(no_of_identities_per_class)

    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             num_workers=num_workers)

    pkl_path = './models/network-conditional.pkl'
    generator = Generator(pkl_path)

    for seed in data_loader:

        generated_images = generator.generate_images(seed, 
                                                     class_index=class_index)

        # You can save or display the generated images as needed

        break 