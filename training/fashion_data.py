import os
import itertools
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from training.utils import image_transform
from llava import conversation as conversation_lib


DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
"""
conv_phi_v0 = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v0",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)
"""
SYSTEM_PROMPT = "A chat between a user with an incomplete outfit and an AI fashion assistant. " \
                "The assistant responds with detailed product descriptions of recommended items to complete the outfit."
instructions = [
    "I have {num_items} items in my outfit. Could you suggest a {category} to complete the look?",
    "Looking for a {category} to go with my {num_items}-piece outfit. What would you recommend?",
    "Need help finding the perfect {category} to complement my current {num_items} items.",
    "With {num_items} pieces in my current outfit, what kind of {category} would work well?",
    "Can you recommend a {category} that would pair nicely with the {num_items} items I'm wearing?",
    "I'm trying to complete my {num_items}-piece look. What {category} would you suggest?",
    "Based on my {num_items} items, what type of {category} should I add to finish the outfit?",
    "Help me choose a {category} that would go well with my current {num_items} pieces.",
    "What {category} would best complete my outfit? I already have {num_items} items picked out.",
    "Considering my {num_items}-piece outfit, could you describe a {category} that would work well?"
]


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


class FashionItemPredictionDataset(Dataset):
    """A dataset class for fashion item prediction from incomplete outfits.

    This dataset handles the task of predicting a fashion item's description based on an incomplete outfit.
    For an outfit with n items, it generates multiple training samples by:
    1. Randomly removing some items to create an incomplete outfit
    2. Selecting one removed item as the ground truth
    3. Using the remaining items as context for prediction

    Attributes:
        data (list): List of outfit dictionaries containing complete outfit information
        tokenizer: Tokenizer for processing text descriptions
        max_length (int): Maximum sequence length for tokenization

    Example:
        For an outfit O={i₁, i₂, i₃, i₄}, possible training samples could be:
        - Input: {i₁, i₂, i₃}, Target: description of i₄
        - Input: {i₁, i₂}, Target: description of i₃
        - Input: {i₄}, Target: description of i₂

    Note:
        Each outfit with n items can generate multiple training samples depending on
        different combinations of incomplete outfits and ground truth selections.
    """
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.transform = image_transform
        image_path = os.path.join(data_path, 'image/291x291')  # Polyvore dataset case

        self.train = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
        self.id_cate_dict = np.load(os.path.join(data_path, "id_cate_dict.npy"), allow_pickle=True).item()
        self.iid_cate_dict = np.load(os.path.join(data_path, "map/iid_cate_dict.npy"), allow_pickle=True).item()
        self.all_image_path = np.load(os.path.join(data_path, "all_item_image_paths.npy"), allow_pickle=True)
        self.all_item_descriptions = np.load(os.path.join(data_path, "all_item_image_descriptions.npy"), allow_pickle=True).item()
        self.samples = []
        for uid, oid, outfit in zip(self.train['uids'], self.train['oids'], self.train["outfits"]):
            # 预先计算常用的映射
            iid_to_cate = {iid: self.iid_cate_dict[iid] for iid in outfit}
            iid_to_cate_str = {iid: self.id_cate_dict[self.iid_cate_dict[iid]] for iid in outfit}
            iid_to_image_path = {iid: os.path.join(image_path, self.all_image_path[iid]) for iid in outfit}
            iid_to_desc = {iid: self.all_item_descriptions[self.all_image_path[iid]] for iid in outfit}
            for n_removes in range(1, len(outfit)):  # 1, 2, 3
                for items_to_remove in itertools.combinations(outfit, n_removes):
                    outfit_copy = set(outfit) - set(items_to_remove)
                    for target_item in items_to_remove:
                        sample = {
                            'uid': uid,
                            # 'oid': oid,
                            # 'incomplete_outfit_id': list(outfit_copy),
                            'incomplete_outfit_path': [iid_to_image_path[iid] for iid in outfit_copy],
                            # 'incomplete_outfit_cate': [iid_to_cate[iid] for iid in outfit_copy],
                            # 'incomplete_outfit_cate_str': [iid_to_cate_str[iid] for iid in outfit_copy],
                            # 'target_id': target_item,
                            # 'target_cate': iid_to_cate[target_item],
                            'target_cate_str': iid_to_cate_str[target_item],
                            # 'target_image_path': iid_to_image_path[target_item],
                            'target_description': iid_to_desc[target_item],
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        for image_path in sample['incomplete_outfit_path']:
            try:
                image = Image.open(image_path).convert('RGB')
                image = image_transform(image)
                images.append(image)
            except:
                print(f"error reading {image_path}")

        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        instruction = random.choice(instructions).format(num_items=len(images), category=sample['target_cate_str'])
        conv.append_message(roles["human"], instruction)





if __name__ == '__main__':
    # from omegaconf import OmegaConf
    # config = OmegaConf.load('/mnt/d/PostDoc/fifth paper/related work/DiFashion/show-o/configs/showo_instruction_tuning_1_w_clip_vit.yaml')
    # preproc_config = config.dataset.preprocessing
    # dataset_config = config.dataset.params
    # total_batch_size_t2i_without_accum = 2
    # dataset = FashionDataset(
    #     root=dataset_config.train_t2i_shards_path_or_url,
    #     caption_file_path=dataset_config.external_caption_path,
    # )
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=4
    # )
    # a = next(iter(dataloader))

    import transformers
    pretrained_model_path = "microsoft/phi-1_5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path,
                                                           padding_side="left")
    special_tokens = ("soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(special_tokens))
    dataset = FashionItemPredictionDataset(
        data_path='/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/polyvore',
        tokenizer=tokenizer
    )
