# -*- coding: utf-8 -*-
"""
Author: 

"""
from gsg.gsg import GSG
from benchmark.cec2013lsgo.cec2013 import Benchmark

benchmark = Benchmark()
fun_id = 1                              #在此处输入函数编号
fun = benchmark.get_function(fun_id)
info = benchmark.get_info(fun_id)

# 根据函数设置参数
if fun_id in [13, 14]:
    info['dimension'] = 905
    info['lower'] = -100
    info['upper'] = 100
    beta = 10000
elif fun_id in [1, 4, 7, 8, 11, 12, 15]:
    info['dimension'] = 1000
    info['lower'] = -100
    info['upper'] = 100
    beta = 10000
elif fun_id in [2, 5, 9]:
    info['dimension'] = 1000
    info['lower'] = -5
    info['upper'] = 5
    beta = 10000
else:  # 3, 6, 10
    info['dimension'] = 1000
    info['lower'] = -32
    info['upper'] = 32
    beta = 1000

gsg = GSG(fun, info)
gsg.beta = beta
subspaces = gsg.run()

print(f"独立变量: {len(subspaces['seps'])}")
print(f"相关组: {len(subspaces['nonseps'])}")



























