import os
from typing import List
import numpy as np
import itertools
import json
from tqdm import tqdm
import math
import hashlib
import time

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import ollama
from transformers import AutoTokenizer, AutoModelForCausalLM


system_message = {
    "role": "system",
    "content": "You are an AI fashion assistant, and you are seeing an incomplete outfit containing several fashion product images given by descriptive sentences.\n"
               "Create diverse questions that seek item recommendations based on the incomplete outfit. "
               "The questions should simulate different user perspectives and scenarios, but all focusing on completing the outfit with a suitable item. "
               "Keep questions natural and conversational, as if someone is looking at the outfit and seeking styling advice, rather than explicitly describing the outfit elements. \n"
               "While formulating questions, incorporate the needed category naturally into the conversation, as how a person would ask for specific item recommendations. \n"
               "When responding, provide recommendations (sentence of item description which will be used for image generation later) and reasoning based solely on analyzing the incomplete outfit, "
               "as if you are discovering the best matching item through fashion expertise rather than knowing the ground truth item. "
               "Your answers should naturally align with the ground truth item's characteristics without explicitly referencing it, "
               "demonstrating authentic fashion reasoning and coordination principles."
}
sample1 = {
    "context": {
        "incomplete_outfit": [
            "This image features a pair of olive green suede ankle boots with an asymmetrical zipper on the outside, showcasing the right boot's heel and toe from a slightly angled perspective while the left boot is shown only from its toe.",
            'This black lace skirt features a delicate fringed hem and an elastic waistband, crafted from intricate floral lace with a subtle sheen that adds to its allure.',
            'This image showcases a delicate white lace top with thin straps, featuring a scalloped edge at the bottom and intricate floral patterns throughout its design.'
        ],
        'ground_truth_item': 'The image presents a green cardigan with two pockets, featuring a front opening and buttons down the center, crafted from an olive-green and brown textured fabric that resembles woven wool.',
        'ground_truth_category': "women's sweater",
    },
    "response": {
        "question": "This lace outfit looks elegant, but I'm concerned about the cooler weather now and need to add a layer. Any suggestions?",
        "answer": "I would suggest adding a soft knit cardigan in an earthy tone like olive green, which would beautifully complement your ankle boots while not overpowering the delicate lace pieces. A wool-blend material would be ideal for warmth and texture. Look for a design with simple buttons down the front and practical pockets – this will add both functionality and sophistication to your ensemble while maintaining the elegant feel of your lace pieces."
    }
}
sample2 = {
    "context": {
        "incomplete_outfit": [
            'This image presents a peach-colored short-sleeved shirt with a round neckline, featuring a small rectangular white label on the inside of the collar. The fabric appears to be either 100% cotton or a blend that mimics its texture. A discreet seam runs along the bottom edge of the garment, adding a touch of subtlety to the overall design.',
            'This image features a black leather jacket with a plush, light brown fur collar and cuffs, showcasing a classic biker-style design with a zip-up front, two side pockets, and a belt at the waist.'
        ],
        'ground_truth_item': "The image features a pair of white jeans with light fading on the thighs and legs, characterized by five belt loops and two back pockets. The front pocket is slightly larger than the back pockets, which are identical in size. A small logo adorns the top right corner of the jeans' waistband.",
        'ground_truth_category': "women's jeans",
    },
    "response": {
        "question": "I'm struggling with what bottoms would balance this look. What would you suggest?",
        "answer": "For this lovely contrast between the soft peach tee and the edgy leather jacket, I'd recommend a pair of light-colored denim, particularly in white or off-white. The lighter wash would create a beautiful balance, complementing the softness of the peach top while providing a clean canvas for the statement leather jacket. Look for a pair with subtle distressing or fading on the thighs to add some character, and classic five-pocket styling would keep it versatile. This would create a perfect blend of casual sophistication that works with both the feminine and edgy elements of your top pieces."
    }
}
sample3 = {
    "context": {
        "incomplete_outfit": [
            'The image features a pair of grey jeans with a brown belt, folded to show the back pocket and stitching details, showcasing their design and quality.'
        ],
        'ground_truth_item': "This image presents a top-down view of a brown leather shoe, likely a Chelsea boot, with an open toe and a red sole. The shoe is centered within the frame, showcasing its upper portion without revealing the heel. Its rich brown hue and smooth texture are evident from this perspective. Notably, the shoe appears to be crafted from high-quality leather, as suggested by its supple appearance.",
        'ground_truth_category': "women's shoe",
    },
    "response": {
        "question": "I've got these jeans that I love, but I'm looking for the perfect shoes to complete the look. Any footwear suggestions that would coordinate well?",
        "answer": "Given the classic grey jeans with the brown belt, I'd recommend a sophisticated leather shoe in a rich brown tone to create a cohesive look by coordinating with your belt. Consider an elevated style like a leather Chelsea boot or an open-toe bootie - something with clean lines and quality leather construction. Look for pieces with interesting details like a distinctive sole that adds character while maintaining versatility. The brown leather would not only complement the belt but also add a polished finish to the grey denim."
    }
}
# working_dir = "/mnt/d/PostDoc/fifth paper/related work/DiFashion"
working_dir = "/workspace/DiFashion"


def format_message(role, content):
    return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"


def generate_sample_id(uid, oid, incomplete_outfit, target_item):
    """根据sample的关键信息生成唯一id"""
    # 组合关键信息
    id_string = f"{uid}_{oid}_{incomplete_outfit}_{target_item}"
    # 使用md5生成固定长度的id
    return hashlib.md5(id_string.encode()).hexdigest()[:16]


def save_checkpoint(results, checkpoint_dir, filename):
    """保存中间结果"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint):
    """加载最新的检查点"""
    if not os.path.exists(checkpoint):
        return [], set()

    with open(checkpoint, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data, set(x['id'] for x in json_data)


class FashionCategories:
    def __init__(self):
        self.essential = {
            "top": [
                "women's chiffon top", "women's t-shirt", "women's sweater", "women's sleeveless top", "women's sweatshirt", "women's shirt",
                "men's polo shirt", "men's sweater", "men's t-shirt", "men's shirt", "men's sweatshirt"
            ],
            "bottom": [
                "women's jeans", "women's casual pants", 'skirt', 'legging',
                "men's jeans", "men's casual pants"
            ],
            "dress": ["dress"],
            "footwear": [
                "women's canvas shoe", "women's slipper", "women's sandal", "women's boot", "women's casual shoe", "women's shoe", 'ankle boot',
                "men's high-top shoe", "men's shoe", 'canvas shoe'
            ],
            "outerwear": [
                "women's wool coat", "women's leather jacket", "women's winter jacket", "women's suit jacket", "women's casual coat",
                "men's jacket", "men's winter jacket", "men's leather jacket", "men's coat", 'trench coat', 'vest'
            ]
        }
        self.accessories = [
            'necklace', 'pendant', 'ring', 'bracelet', 'bangle', 'earrings', 'belt', 'hat', 'watch', 'travel bag', 'crossbody bag'
        ]

        # 创建一个扁平化的所有商品列表，用于快速检查
        self._all_essential_categories = set()
        for categories in self.essential.values():
            self._all_essential_categories.update(categories)

    def is_essential(self, x):
        return x in self._all_essential_categories

    def get_category(self, x):
        # 获取商品所属的类别
        if x in self.accessories:
            return "accessory"
        for category, sub_categories in self.essential.items():
            if x in sub_categories:
                return category
        return None

    def is_accessory(self, x):
        return x in self.accessories


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
        image_path = os.path.join(data_path, 'image/291x291')  # Polyvore dataset case
        self.fashion_categories = FashionCategories()

        self.train = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
        self.id_cate_dict = np.load(os.path.join(data_path, "id_cate_dict.npy"), allow_pickle=True).item()
        self.iid_cate_dict = np.load(os.path.join(data_path, "map/iid_cate_dict.npy"), allow_pickle=True).item()
        self.all_image_path = np.load(os.path.join(data_path, "all_item_image_paths.npy"), allow_pickle=True)
        self.all_item_descriptions = np.load(os.path.join(data_path, "all_item_image_descriptions.npy"), allow_pickle=True).item()
        self.samples = []
        # for uid, oid, outfit in itertools.islice(zip(self.train['uids'], self.train['oids'], self.train["outfits"]), 20):
        for uid, oid, outfit in zip(self.train['uids'], self.train['oids'], self.train["outfits"]):
            # 预先计算常用的映射
            iid_to_cate = {iid: self.iid_cate_dict[iid] for iid in outfit}
            iid_to_cate_str = {iid: self.id_cate_dict[self.iid_cate_dict[iid]] for iid in outfit}
            iid_to_image_path = {iid: self.all_image_path[iid] for iid in outfit}
            iid_to_desc = {iid: self.all_item_descriptions[self.all_image_path[iid]] for iid in outfit}
            for n_removes in range(1, len(outfit)):  # 1, 2, 3
                for items_to_remove in itertools.combinations(outfit, n_removes):
                    outfit_copy = set(outfit) - set(items_to_remove)
                    for target_item in items_to_remove:
                        incomplete_outfit_cate_str = [iid_to_cate_str[iid] for iid in outfit_copy]
                        target_image_path = iid_to_image_path[target_item]
                        sample = {
                            'id': generate_sample_id(uid, oid, list(outfit_copy), target_item),
                            'uid': uid,
                            'oid': oid,
                            # 'incomplete_outfit_id': list(outfit_copy),
                            'incomplete_outfit_path': [iid_to_image_path[iid] for iid in outfit_copy],
                            'incomplete_outfit_description': [iid_to_desc[iid] for iid in outfit_copy],
                            # 'incomplete_outfit_cate': [iid_to_cate[iid] for iid in outfit_copy],
                            # 'incomplete_outfit_cate_str': [iid_to_cate_str[iid] for iid in outfit_copy],
                            # 'target_id': target_item,
                            # 'target_cate': iid_to_cate[target_item],
                            'target_cate_str': iid_to_cate_str[target_item],
                            'target_image_path': target_image_path,
                            'target_description': iid_to_desc[target_item],
                            'weight': self._calculate_weight(incomplete_outfit_cate_str, len(items_to_remove))
                        }
                        if sample['weight'] > 0.0:
                            self.samples.append(sample)

    def _calculate_weight(self, incomplete_outfit_cate_str: List[str], n_removes: int):
        """
        计算每个样本的权重

        Args:
            incomplete_outfit_cate_str: 不完整搭配的类别列表
            n_removes: 移除的商品数量
        """
        original_len = len(incomplete_outfit_cate_str) + n_removes

        # 1. 计算基础权重
        # current: 1.0, 0.816, 0.577
        weight = round(math.pow((original_len - n_removes) / (original_len - 1), 0.5), 3)

        # 2. 如果incomplete里面没有核心类别，则置系数为0， 即不能出现用耳环推荐鞋子这种情况
        if sum([self.fashion_categories.is_essential(x) for x in incomplete_outfit_cate_str]) == 0:
            weight *= 0.0

        return weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # sample = self.samples[idx]
        # prompt = format_message("system", system_message["content"])
        # for example in [sample1, sample2, sample3]:
        #     prompt += "\n" + format_message("user", json.dumps(example["context"], indent=2))
        #     prompt += "\n" + format_message("assistant", json.dumps(example["response"], indent=2))
        #
        # context = {
        #     "incomplete_outfit": sample['incomplete_outfit_description'],
        #     'ground_truth_item': sample['target_description'],
        #     'ground_truth_category': sample['target_cate_str']
        # }
        # prompt += "\n" + format_message("user", json.dumps(context, indent=2))
        # inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        # return prompt, inputs
        return self.samples[idx]


if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     "microsoft/phi-4",
    #     trust_remote_code=True,
    #     torch_dtype=torch.float16,  # 使用半精度来减少内存使用
    #     device_map="auto"  # 自动处理设备映射
    # )
    # dataset = FashionItemPredictionDataset(
    #     data_path='/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/polyvore',
    #     tokenizer=tokenizer
    # )
    # prompt, inputs = dataset[0]
    # outputs = model.generate(
    #     **inputs.to('cuda')
    # )
    # responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(responses)

    dataset = FashionItemPredictionDataset(
        data_path=os.path.join(working_dir, "datasets/polyvore"),
        tokenizer=None
    )

    # 加载检查点
    result_output_path = os.path.join(working_dir, "datasets/polyvore/instruct", "fashion_recommendation_qa.json")
    results, processed_ids = load_checkpoint(result_output_path)

    # 创建错误日志文件
    error_log_path = os.path.join(working_dir, "datasets/polyvore/instruct/error_log.json")
    if not os.path.exists(error_log_path):
        errors = []
    else:
        with open(error_log_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        errors = [x['id'] for x in json_data]

    batch_size = 5000
    start_time = time.time()

    for i, sample in enumerate(tqdm(dataset)):
        sample_id = sample["id"]
        if sample_id in processed_ids and sample_id not in errors:
            continue
        context = {
            "incomplete_outfit": sample['incomplete_outfit_description'],
            'ground_truth_item': sample['target_description'],
            'ground_truth_category': sample['target_cate_str']
        }
        response = ollama.chat(
            model='phi4',
            messages=[
                system_message,
                {
                    'role': 'user',
                    'content': json.dumps(sample1['context'], indent=2),
                },
                {
                    'role': 'assistant',
                    'content': json.dumps(sample1['response'], indent=2),
                },
                {
                    'role': 'user',
                    'content': json.dumps(sample2['context'], indent=2),
                },
                {
                    'role': 'assistant',
                    'content': json.dumps(sample2['response'], indent=2),
                },
                {
                    'role': 'user',
                    'content': json.dumps(sample3['context'], indent=2),
                },
                {
                    'role': 'assistant',
                    'content': json.dumps(sample3['response'], indent=2),
                },
                {
                    'role': 'user',
                    'content': json.dumps(context, indent=2),
                },
            ]
        )
        try:
            x = json.loads(response["message"]["content"])
            question = x["question"]
            answer = x["answer"]
            result = {
                "id": sample["id"],
                "uid": sample["uid"],
                "oid": sample["oid"],
                "incomplete_outfit_path": sample['incomplete_outfit_path'],
                'target_cate_str': sample['target_cate_str'],
                'target_image_path': sample['target_image_path'],
                'target_description': sample['target_description'],
                'weight': sample['weight'],
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            results.append(result)
            processed_ids.add(sample["id"])
        except Exception as e:
            # 记录错误
            error_info = {
                "id": sample_id,
                "error": str(e),
                "sample": {
                    "uid": sample["uid"],
                    "oid": sample["oid"],
                    "target_image_path": sample['target_image_path']
                }
            }
            errors.append(error_info)

        # 每处理batch_size个样本保存一次
        if (i + 1) % batch_size == 0:
            save_checkpoint(results, os.path.join(working_dir, "datasets/polyvore/instruct"), "fashion_recommendation_qa.json")

            # 保存错误日志
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)

            # 打印进度
            elapsed_time = time.time() - start_time
            print(f"\nProcessed {i + 1} samples in {elapsed_time:.2f} seconds")
            print(f'Results saved to {os.path.join(working_dir, "datasets/polyvore/instruct", "fashion_recommendation_qa.json")}')
            print(f"Errors saved to {error_log_path}")

    # 保存最终结果
    with open(result_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存最终错误日志
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing completed!")
    print(f"Total samples processed: {len(results)}")
    print(f"Total errors: {len(errors)}")
    print(f"Final results saved to: {result_output_path}")
    print(f"Error log saved to: {error_log_path}")
