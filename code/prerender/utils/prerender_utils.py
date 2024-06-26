import tensorflow as tf
import os
import numpy as np
from .vectorizer import MultiPathPPRenderer
from .utils import get_config, data_to_numpy
import argparse

def get_visualizer(renderer_name, renderer_config):
    if renderer_name == "MultiPathPPRenderer":
        return MultiPathPPRenderer(renderer_config)
    raise Exception(f"Unknown visualizer {renderer_name}")

def get_visualizers(visualizers_config):
    visualizers = []
    for renderer in visualizers_config:
        visualizers.append(get_visualizer(renderer["renderer_name"], renderer["renderer_config"]))
    return visualizers

def create_dataset(datapath, n_shards, shard_id):       #作用：创建dataset
    files = os.listdir(datapath)                        #返回指定目录下的所有文件及文件夹名称
    '''
    https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    继承自dataset: https://zhuanlan.zhihu.com/p/30751039
    同时支持从内存(原placeholder)和硬盘(原queue)中读取数据
    dataset是“元素”的有序列表，示例化一个Iterator，然后对Iterator进行迭代
    '''
    dataset = tf.data.TFRecordDataset(                  
        [os.path.join(datapath, f) for f in files], num_parallel_reads=1
    )
    if n_shards > 1:
        dataset = dataset.shard(n_shards, shard_id)
    return dataset

def parse_arguments(): #作用：解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw data")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save data")
    parser.add_argument("--n-jobs", type=int, default=20, required=False, help="Number of threads")
    parser.add_argument(
        "--n-shards", type=int, default=8, required=False, help="Use `1/n_shards` of full dataset")
    parser.add_argument(
        "--shard-id", type=int, default=0, required=False, help="Take shard with given id")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    return args

def generate_filename(scene_data):
    scenario_id = scene_data["scenario_id"]
    agent_id = scene_data["agent_id"]
    agent_type = scene_data["target/agent_type"]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"

def merge_and_save(visualizers, data, output_path): #作用：将数据转换为numpy格式并保存
    data_to_numpy(data)
    preprocessed_dicts = [visualizer.render(data) for visualizer in visualizers]
    for scene_number in range(len(preprocessed_dicts[0])):
        scene_data = {}
        for visualizer_number in range(len(preprocessed_dicts)):
            scene_data.update(preprocessed_dicts[visualizer_number][scene_number])
        np.savez_compressed(os.path.join(output_path, generate_filename(scene_data)), **scene_data)