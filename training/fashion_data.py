import os

from PIL import Image
from training.utils import image_transform
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FashionDataset(Dataset):
    def __init__(
            self,
            root: str,
            caption_file_path: str,
            image_size=256,
    ):
        self.root = root
        self.captions = np.load(caption_file_path, allow_pickle=True).item()
        self.captions = [(k, v) for k, v in self.captions.items() if k != "empty_image.png"]  # remove empty image
        self.transform = image_transform
        self.image_size = image_size

    def __getitem__(self, idx):
        try:
            image_path, caption = self.captions[idx]

            image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
            image = self.transform(image, resolution=self.image_size)

            return {'images': image, 'input_ids': caption}

        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1)

    def __len__(self):
        return len(self.captions)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    config = OmegaConf.load('/mnt/d/PostDoc/fifth paper/related work/DiFashion/show-o/configs/showo_instruction_tuning_1_w_clip_vit.yaml')
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    total_batch_size_t2i_without_accum = 2
    dataset = FashionDataset(
        root=dataset_config.train_t2i_shards_path_or_url,
        caption_file_path=dataset_config.external_caption_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    a = next(iter(dataloader))
