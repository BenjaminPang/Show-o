import os
import json
from functools import partial

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torch

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
SYSTEM_PROMPT = "As a fashion assistant, you recommend suitable items with detailed descriptions to help users complete their outfits effectively."


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


def preprocess_v0(
        sources,
        tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )


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
        self.image_dir = os.path.join(data_path, 'image/291x291')  # Polyvore dataset case
        with open(os.path.join(data_path, 'instruct/fashion_recommendation_qa.json')) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        for image_filename in sample['incomplete_outfit_path']:
            try:
                image = Image.open(os.path.join(self.image_dir, image_filename)).convert('RGB')
                image = image_transform(image)
                images.append(image)
            except:
                print(f"error reading {os.path.join(self.image_dir, image_filename)}")

        data_dict = preprocess_v0([sample["conversations"]], self.tokenizer)

        if isinstance(idx, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0],
                             images=torch.stack(images))

        return data_dict


def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                for key in ("input_ids", "labels", "input_ids_system"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    input_ids_system = torch.stack(input_ids_system, dim=0)

    offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

    if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(
        max_length - input_ids_system.shape[-1],
        tokenizer.model_max_length - input_ids_system.shape[-1],
    )

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        input_ids_system=input_ids_system,
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    return batch


class FashionItemPredictionDataloader:
    def __init__(self, tokenizer, batch_size, num_workers, world_size, local_rank, max_length, phase):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.local_rank = local_rank
        self.max_length = max_length
        self.phase = phase

    def get_dataloader(self):
        train_dataset = FashionItemPredictionDataset(
            self.tokenizer,
            self.phase,
        )
        datasampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank
        )

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            ),
            sampler=datasampler
        )

        return dataloader


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
    # dataset = FashionItemPredictionDataset(
    #     data_path='/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/polyvore',
    #     tokenizer=tokenizer
    # )
    # x = dataset[0]
    dataloader = FashionItemPredictionDataloader(
        tokenizer,
        batch_size=2,
        num_workers=0,
        world_size=1,
        local_rank=0,
        max_length=1024,
        phase="tuning"
    )
    x = next(iter(dataloader))
    a = 1
