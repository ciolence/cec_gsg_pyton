# -*- coding: utf-8 -*-
"""
Author: 

"""
# gsg_exp.py (主程序文件)
from gsg.gsg import GSG
from benchmark.cec2013lsgo.cec2013 import Benchmark
import time
import numpy as np


def main():
    # 初始化基准测试集
    benchmark = Benchmark()

    # 遍历1到15号测试函数
    for fun_id in range(1, 16):
        print(f"\n{'=' * 60}")
        print(f"处理函数 {fun_id}/15")
        print(f"{'=' * 60}")

        start_time = time.time()

        # 获取当前测试函数及其信息
        fun = benchmark.get_function(fun_id)
        info = benchmark.get_info(fun_id)

        # 根据论文设置不同函数的参数
        t1 = [13, 14]
        t2 = [1, 4, 7, 8, 11, 12, 15]
        t3 = [2, 5, 9]  # 3,6,10

        if fun_id in t1:
            info['dimension'] = 905
            info['lower'] = -100
            info['upper'] = 100
            beta = 10000
        elif fun_id in t2:
            info['dimension'] = 1000
            info['lower'] = -100
            info['upper'] = 100
            beta = 10000
        elif fun_id in t3:
            info['dimension'] = 1000
            info['lower'] = -5
            info['upper'] = 5
            beta = 10000
        else:  # 3, 6, 10
            info['dimension'] = 1000
            info['lower'] = -32
            info['upper'] = 32
            beta = 1000

        # 初始化GSG算法
        gsg = GSG(fun, info)
        gsg.beta = beta
        gsg.alpha = 0.0  # 误差阈值系数

        # 运行算法，获取变量分组结果
        subspaces = gsg.run()

        # 输出结果
        print(f"函数 {fun_id} 结果:")
        print(f"  独立变量数: {len(subspaces['seps'])}")
        print(f"  相关变量组数: {len(subspaces['nonseps'])}")

        # 打印相关变量组大小分布
        group_sizes = [len(group) for group in subspaces['nonseps']]
        if group_sizes:
            print(f"  相关组大小分布: 最小{min(group_sizes)}, 最大{max(group_sizes)}, 平均{np.mean(group_sizes):.2f}")

        print(f"  函数评估次数: {subspaces['FEs']}")
        print(f"  GSS函数评估: {subspaces['GSS_FEs']}")
        print(f"  超出边界次数: {subspaces['outBoundTimes']}")
        print(f"  运行时间: {time.time() - start_time:.2f}秒")

        # 保存结果到文件
        save_results(fun_id, subspaces)


def save_results(fun_id, subspaces):
    """保存结果到文件"""
    import pickle
    import os

    # 创建结果目录
    if not os.path.exists('./results'):
        os.makedirs('./results')

    filename = f'./results/F{fun_id:02d}.pkl'

    # 保存所有信息
    with open(filename, 'wb') as f:
        pickle.dump(subspaces, f)

    # 也保存为文本格式便于查看
    txt_filename = f'./results/F{fun_id:02d}_summary.txt'
    with open(txt_filename, 'w') as f:
        f.write(f"Function ID: {fun_id}\n")
        f.write(f"Independent variables (seps): {len(subspaces['seps'])}\n")
        f.write(f"Dependent groups (nonseps): {len(subspaces['nonseps'])}\n")
        f.write(f"Total FEs: {subspaces['FEs']}\n")
        f.write(f"GSS FEs: {subspaces['GSS_FEs']}\n")
        f.write(f"Out of bound times: {subspaces['outBoundTimes']}\n")
        f.write(f"Max detection step: {subspaces['MaxDetectionStep']}\n")
        f.write("\nIndependent variables:\n")
        f.write(str(subspaces['seps']) + '\n')
        f.write("\nDependent groups:\n")
        for i, group in enumerate(subspaces['nonseps']):
            f.write(f"Group {i + 1} (size={len(group)}): {group}\n")

    print(f"  结果已保存到: {filename}")


if __name__ == "__main__":
    # 运行主程序
    total_start = time.time()
    main()
    print(f"\n总运行时间: {time.time() - total_start:.2f}秒")


























