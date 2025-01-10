import copy
import random
import os
from functools import partial

import numpy as np
import torch
from PIL import Image
from llava import conversation as conversation_lib
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPImageProcessor

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]


def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_plain(sources, tokenizer):
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        # source[0]['value'] = DEFAULT_IMAGE_TOKEN
        source[0]['value'] = ""
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    # input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    input_ids = [tokenizer(prompt)["input_ids"] + [tokenizer.eos_token_id] for prompt in conversations]
    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        # tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        tokenized_len = len(tokenizer(source[0]['value'])["input_ids"])
        if tokenized_len > 0:
            target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(targets))


class LLaVAPretrainCaptioningDataset(Dataset):

    def __init__(self, tokenizer, data_file_path, image_root):
        super(LLaVAPretrainCaptioningDataset, self).__init__()

        self.tokenizer = tokenizer

        self.image_root = image_root
        brief_description_prompts = [
            "Describe the image concisely.",
            "Provide a brief description of the given image.",
            "Offer a succinct explanation of the picture presented.",
            "Summarize the visual content of the image.",
            "Give a short and clear explanation of the subsequent image.",
            "Share a concise interpretation of the image provided.",
            "Present a compact description of the photo's key features.",
            "Relay a brief, clear account of the picture shown.",
            "Render a clear and concise summary of the photo.",
            "Write a terse but informative summary of the picture.",
            "Create a compact narrative representing the image presented."
        ]

        data = np.load(data_file_path, allow_pickle=True).item()
        self.list_data_dict = []
        for image_filename, description in data.items():
            if image_filename == 'empty_image.png':  # First image is an empty image, so we remove it
                continue
            sample = {
                "image": image_filename,
                "conversations": [
                    {"from": "human", "value": random.choice(brief_description_prompts)},
                    {"from": "gpt", "value": description}
                ]
            }
            self.list_data_dict.append(sample)

        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        print("Formatting llava captioning data")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_root
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # This operation manually remove <image> token from value.
        # Since we didn't add it in source data, we do not need to invoke preprocess_multimodal function.
        # sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_plain(sources, self.tokenizer)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict


def collate_fn(
        instances,
        tokenizer=None,
        max_length=77,
):
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)

    if input_ids.shape[-1] < max_length:
        offset = max_length - input_ids.shape[-1]
        pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, pad_tube], dim=1)

        offset = max_length - labels.shape[-1]
        pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        labels = torch.cat([labels, pad_tube], dim=1)

    min_max_len = min(max_length, tokenizer.model_max_length)

    input_ids = input_ids[:, :min_max_len]
    labels = labels[:, :min_max_len]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    return batch


def get_plain_data_loader(
        tokenizer,
        data_file_path,
        image_root,
        batch_size,
        num_workers,
        world_size,
        local_rank,
        max_length,
):
    train_dataset = LLaVAPretrainCaptioningDataset(tokenizer, data_file_path, image_root)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader


if __name__ == '__main__':
    import transformers
    pretrained_model_path = "microsoft/phi-1_5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path,
                                                           padding_side="left")
    special_tokens = ("soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(special_tokens))

    train_dataloader_mmu = get_plain_data_loader(
        tokenizer,
        batch_size=10,
        num_workers=0,
        world_size=1,
        local_rank=0,
        max_length=512 - (576 - 256),
    )
    a = next(iter(train_dataloader_mmu))
    x = 1
