import multiprocessing
from tqdm import tqdm
import tensorflow as tf
from utils.features_description import generate_features_description
from utils.prerender_utils import (create_dataset, get_visualizers, merge_and_save, parse_arguments)
from utils.utils import get_config

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    args = parse_arguments()  #输入命令行参数： 
    visualizers_config = get_config(args.config)
    visualizers = get_visualizers(visualizers_config)
    #将每个tfrecord文件的渲染结果分别存放
    for f in sorted(os.listdir(args.data_path)):
        print("# Processing file:", os.path.join(args.data_path, f))

        output_path = os.path.join(args.output_path, f[-14:-9])
        print("# Output path:", output_path)
        # 文件夹不存在会报错
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        dataset = tf.data.TFRecordDataset([os.path.join(args.data_path, f)], 
            num_parallel_reads=1)
        
        p = multiprocessing.Pool(args.n_jobs) #创建 n_jobs个子进程
        processes = []
        k = 0
        for data in tqdm(dataset.as_numpy_iterator()):
            k += 1
            data = tf.io.parse_single_example(data, generate_features_description())
            processes.append(
                p.apply_async( #apply会阻塞主进程并按顺序执行子进程，apply_async异步执行子进程，传入函数名+参数列表
                    merge_and_save,
                    kwds=dict(
                        visualizers=visualizers,
                        data=data,
                        output_path=output_path,
                    ),
                )
            )

        for r in tqdm(processes):
            r.get()

if __name__ == "__main__":
    main()
