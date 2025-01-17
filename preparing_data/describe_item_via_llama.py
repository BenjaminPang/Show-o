import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import ollama
import time


def load_checkpoint():
    """加载检查点文件，如果存在的话"""
    checkpoint_file = os.path.join('temp', "image_processing_checkpoint.npy")
    if os.path.exists(checkpoint_file):
        return np.load(checkpoint_file, allow_pickle=True).item()
    return {'empty_image.png': "empty image"}


def save_checkpoint(output_file, checkpoint_number):
    """保存检查点和完整结果"""

    # 保存当前检查点
    checkpoint_file = os.path.join('temp', "image_processing_checkpoint.npy")
    np.save(checkpoint_file, output_file)

    # 保存带编号的备份
    backup_file = os.path.join('temp', f"image_processing_backup_{checkpoint_number}.npy")
    np.save(backup_file, output_file)
    print(f"Saved checkpoint and backup #{checkpoint_number}")


def get_last_processed_index(output_file, all_item_image_path):
    """获取上次处理到的索引"""
    processed_images = set(output_file.keys())
    for i, image_path in enumerate(all_item_image_path):
        if image_path not in processed_images:
            return max(1, i)  # 确保至少从索引1开始（跳过empty_image）
    return len(all_item_image_path)


if __name__ == "__main__":
    # 设置输入和备份文件夹路径
    input_root = "image/291x291"

    # 创建空白图片
    empty_img = Image.new('RGB', (512, 512), (255, 255, 255))
    empty_img.save(os.path.join(input_root, "empty_image.png"))

    # 加载图片路径
    all_item_image_path = np.load('../../datasets/polyvore/all_item_image_paths.npy', allow_pickle=True)
    image_dir = '../../datasets/polyvore/image/291x291'
    prompt = (
        "Describe this fashion product image in ONE sentence. DO NOT mention background or camera angle."
    )

    # 加载检查点或创建新的输出文件
    output_file = load_checkpoint()

    # 获取上次处理到的位置
    start_index = get_last_processed_index(output_file, all_item_image_path)

    # 计算总批次数
    checkpoint_size = 1000
    total_images = len(all_item_image_path[start_index:])

    print(f"Resuming from image index {start_index}")

    try:
        for i, image_path in enumerate(tqdm(all_item_image_path[start_index:]), start=start_index):
            try:
                # 检查是否已处理过该图片
                if image_path not in output_file:
                    response = ollama.chat(
                        model='llama3.2-vision',
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [os.path.join(image_dir, image_path)]
                        }]
                    )
                    output_file[image_path] = response["message"]["content"]

                # 每处理checkpoint_size张图片保存一次
                if (i - start_index + 1) % checkpoint_size == 0:
                    save_checkpoint(output_file, i)

                # 添加小延时避免请求过于频繁
                time.sleep(0.1)

            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
                # 发生错误时也保存检查点
                save_checkpoint(output_file, i)
                continue

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving checkpoint...")
        save_checkpoint(output_file, i)

    # 最终保存完整结果
    np.save("../../datasets/polyvore/all_item_image_descriptions.npy", output_file)
    print("Processing completed!")
