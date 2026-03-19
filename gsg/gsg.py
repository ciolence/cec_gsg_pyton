# -*- coding: utf-8 -*-
"""
Author: cls

"""
# gsg.py
import numpy as np
import time
from tqdm import tqdm


class GSG:
    def __init__(self, fun, info):
        """
        GSG算法初始化

        :param fun: 目标函数 (可调用对象)
        :param info: 字典, 包含 'dimension'(维度), 'lower'(下界), 'upper'(上界)
        """
        self.fun = fun  # 目标函数
        self.info = info    # 函数信息字典, 包含:
                            #   - 'dimension': 维度
                            #   - 'lower': 下界
                            #   - 'upper': 上界

        # GSG算法参数
        self.epsilon = 1e-6 # 数值精度阈值，用于黄金分割法和收敛判断
        self.alpha = 0.0  # 误差阈值系数，用于计算交互检测的误差阈值
        self.beta = 10000   # 检测步长缩放因子，detectionStep = epsilon * beta
        self.plot1D = False # 是否可视化

        # 全局变量初始化
        self.A_FEs = 0  # 总函数评估计数，A代表Advance_Retreat
        self.GSS_FEs = 0    # 黄金分割法评估次数
        self.rootVar = None # 当前检测变量的索引
        self.detectingGroup = None  # 当前检测组
        self.cv = None  # 当前点，D维向量
        self.perturb = None # 扰动值
        self.isPerturb = False  #是否进行扰动

        # interactionDetection中的变量
        self.outBoundFlag = 0   # 是否超出边界
        self.outBoundTimes = 0  # 超出边界的次数
        self.MaxDetectionStep = 0   # 最大检测步长
        self.MaxDeSTimes = 0    #检测步长大于1的次数

        # GeneralSeparabilityDetection中的变量
        self.turnTransformation = 1 # 是否开启变换
        self.turnImbalance = 1  # 是否开启不平衡处理
        self.A_localOptALL = None   # 局部最优值储存
        self.cvAll = None   # 历史点储存，矩阵，形状(D, D)，第i行，当变量i是rootVar时，对应的cv值

    def run(self):
        """
        GSG算法主流程 - 一般可分离性检测

        返回: 包含'seps'(独立变量)和'nonseps'(相关变量组)的字典
        """
        # 重置计数器
        self.A_FEs = 0
        self.GSS_FEs = 0

        D = self.info['dimension']
        lb = self.info['lower']
        ub = self.info['upper']

        # 设置参数
        eps = self.epsilon
        alpha = self.alpha
        detectionStep = eps * self.beta
        optError = ub / 1000    # 最优误差阈值
                                # 用于判断最优值是否在边界附近

        opts = {
            'lbound': lb,
            'ubound': ub,
            'dim': D
        }

        allVars = list(range(D))    # 所有变量索引列表 [0, 1, 2, ... , D-1]

        # 初始化存储
        A__fullySepGroup = []  # 完全可分变量列表（seps）
        A__partSepGroups = []  # 部分可分（相关）变量组列表（nonseps）

        # 初始化当前点和所有局部最优
        self.cv = np.zeros(D)
        self.cvAll = np.zeros((D, D))
        self.A_localOptALL = np.zeros(D)

        # 设置扰动
        self.perturb = ub

        print(f"开始检测维度 {D} 的变量分组...")

        # 第一步：为每个变量找到局部最优
        print("第一步：为每个变量找局部最优...")
        for i in tqdm(range(D - 1, -1, -1), desc="变量局部优化"):
            self.rootVar = i
            A_localOpt = self.advance_retreat_gold(func_num=1, t0=0, step=detectionStep,
                                                   eps=eps, opts=opts)[0]
            self.A_localOptALL[i] = A_localOpt

            # 如果最优值在边界附近，则重新设为中点
            if abs(abs(A_localOpt) - ub) <= optError:
                self.A_localOptALL[i] = (ub + lb) / 2

            self.cv[self.rootVar] = self.A_localOptALL[i]
            self.cvAll[self.rootVar, :] = self.cv.copy()

        # 记录函数评估次数
        self.GSS_FEs = self.A_FEs

        # 重新计算在边界的变量的最优值
        for i in range(D - 1, -1, -1):
            if self.A_localOptALL[i] == (ub + lb) / 2:
                self.rootVar = i
                A_localOpt = self.advance_retreat_gold(func_num=1, t0=self.A_localOptALL[i],
                                                       step=detectionStep, eps=eps, opts=opts)[0]
                self.A_localOptALL[self.rootVar] = A_localOpt
                self.cv[self.rootVar] = A_localOpt
                self.cvAll[self.rootVar, :] = self.cv.copy()

        # 第二步：交互检测
        print("第二步：交互检测...")
        outBoundVars = []   # 超出边界的变量索引列表（需要重新检测）

        for rootVar in tqdm(range(D), desc="交互检测"):
            group = [rootVar]   # 当前正在构建的相关变量组
            pickStackVars = 0

            # 检查是否已有组包含此变量
            for j, part_group in enumerate(A__partSepGroups):
                if rootVar in part_group:  # 合并间接组
                    groupStack = [list(set(allVars) - set(part_group))]     # 列表的列表，储存待检测的变量组，LIFO
                    pickStackVars = 1
                    break

            if pickStackVars == 0:
                groupStack = [list(set(allVars) - {rootVar})]

            # 设置当前点和局部最优
            self.cv = self.cvAll[rootVar, :].copy()
            A_localOpt = self.A_localOptALL[rootVar]
            self.rootVar = rootVar

            # 处理边界变量
            if rootVar in outBoundVars:
                self.outBoundFlag = 1

            # 内部循环：检测变量与组的关系
            while groupStack:
                if self.outBoundFlag == 1:
                    break

                detectingGroup = groupStack[-1]
                groupStack = groupStack[:-1]  # 弹出最后一个

                # 计算扰动
                perturbTemp = self.cv[detectingGroup].copy()
                perturbTemp = -np.sign(perturbTemp)
                perturbTemp[perturbTemp == 0] = 1
                self.perturb = perturbTemp * ub
                self.detectingGroup = detectingGroup

                # 执行交互检测
                isInteract, A_localOpt_P, detectionStepFixed, A_deltaLeft, A_deltaRight = self.interaction_detection(
                    rootVar=rootVar,
                    detectingGroup=detectingGroup,
                    A_localOpt=A_localOpt,
                    detectionStep=detectionStep,
                    alpha=alpha,
                    func_num=1,
                    eps=eps,
                    opts=opts
                )

                if self.outBoundFlag == 1:
                    break

                # 处理检测结果
                if isInteract == 1:  # 有交互
                    if len(detectingGroup) == 1:
                        group.extend(detectingGroup)
                    else:
                        # 对半分组
                        splitIndex = len(detectingGroup) // 2
                        groupStack.append(detectingGroup[splitIndex:])
                        groupStack.append(detectingGroup[:splitIndex])
                # else: 无交互，不处理

                # 可视化（如果启用）
                if self.plot1D:
                    self._plot_1D(rootVar, detectingGroup, lb, ub, func_num=1)

            # 处理边界条件
            if self.outBoundFlag == 1:
                outBoundVars.append(rootVar)
                self.outBoundFlag = 0
                continue

            # 处理检测结果
            if len(group) > 1:
                merging_indices = []

                # 查找需要合并的组
                for j, part_group in enumerate(A__partSepGroups):
                    if any(var in part_group for var in group):
                        merging_indices.append(j)

                if merging_indices:
                    # 合并到第一个组
                    new_group = set()
                    for idx in merging_indices:
                        new_group.update(A__partSepGroups[idx])
                    new_group.update(group)

                    # 保留第一个，删除其他
                    A__partSepGroups[merging_indices[0]] = list(new_group)
                    for idx in sorted(merging_indices[1:], reverse=True):
                        A__partSepGroups.pop(idx)
                else:
                    # 新组
                    A__partSepGroups.append(group)

        # 计算完全可分变量
        A__fullySepGroup = list(set(allVars))
        for part_group in A__partSepGroups:
            A__fullySepGroup = [var for var in A__fullySepGroup if var not in part_group]

        # 整理输出格式
        FEs = self.A_FEs
        epsilon = eps

        subspaces = {
            'seps': A__fullySepGroup,
            'nonseps': A__partSepGroups,
            'FEs': FEs,
            'GSS_FEs': self.GSS_FEs,
            'epsilon': epsilon,
            'beta': self.beta,
            'A_localOptALL': self.A_localOptALL,
            'outBoundTimes': self.outBoundTimes,
            'MaxDetectionStep': self.MaxDetectionStep,
            'MaxDeSTimes': self.MaxDeSTimes
        }

        return subspaces

    def _plot_1D(self, rootVar, detectingGroup, lb, ub, func_num=1, n_points=1000):
        """
        绘制1D可视化（可选功能）
        """
        import matplotlib.pyplot as plt

        xAxis = np.linspace(lb, ub, n_points)

        # 原始曲线
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        YLine_orig = []
        for x in xAxis:
            solution = self.cv.copy()
            solution[rootVar] = x
            y = self.fun(solution)
            YLine_orig.append(y)
        plt.plot(xAxis, YLine_orig)
        plt.title('Original')

        # 扰动曲线
        plt.subplot(2, 1, 2)
        YLine_pert = []
        for x in xAxis:
            solution = self.cv.copy()
            solution[rootVar] = x
            solution[detectingGroup] = self.perturb[:len(detectingGroup)]
            y = self.fun(solution)
            YLine_pert.append(y)
        plt.plot(xAxis, YLine_pert)
        plt.title('Perturbed')

        plt.tight_layout()
        plt.show()

    # 之前已经实现的函数
    def advance_retreat_gold(self, func_num, t0, step, eps, opts):
        """
        进退法和黄金分割法确定极小值

        输入：
        func_num: 函数编号
        t0: 进退法端点值
        step: 进退法步长
        eps: 黄金分割法的精度
        opts: 选项字典，包含 lbound, ubound

        输出：
        localOpt: 找到的最优值
        Monotonicity: 单调性 (0: 找到最优, 1: 递增, -1: 递减)
        """
        # 对应Matlab中的Advance_Retreat_Gold主函数
        startPoint = opts.get('lbound', t0)
        endPoint = opts.get('ubound', t0 + 10 * step)

        # 如果提供了边界，使用边界
        if 'lbound' in opts and 'ubound' in opts:
            startPoint = opts['lbound']
            endPoint = opts['ubound']
        else:
            # 否则使用进退法确定区间
            endPoint, startPoint, monot = self._opt_advance_retreat(func_num, t0, step, opts)
            if abs(endPoint - t0) < eps:
                print('此端点右侧单调递增，没有搜索到极小值区间，请修改端点值!')
                return np.nan, monot

        # 黄金分割法确定精确解
        localOpt = self._opt_gold(func_num, startPoint, endPoint, eps)
        return localOpt, 0

    def _opt_advance_retreat(self, func_num, t0, step, opts):
        """
        进退法确定最优解区间

        输入：
        func_num: 函数编号
        t0: 进退法端点值
        step: 进退法步长
        opts: 选项字典

        输出：
        endPoint: 区间右端点
        startPoint: 区间左端点
        monot: 单调性标志
        """
        t1 = t0
        t2 = t1 + step
        t3 = t2 + step

        # 评估函数值
        ft0 = self._evaluation(func_num, t0)
        ft1 = ft0
        ft2 = self._evaluation(func_num, t2)
        ft3 = self._evaluation(func_num, t3)

        startPoint = t1
        endPoint = t1
        monot = 0

        ubound = opts.get('ubound', float('inf'))

        while t3 + step <= ubound:
            if (ft2 <= ft1) and (ft2 <= ft3):  # 找到最优区间
                startPoint = t1
                endPoint = t3
                return endPoint, startPoint, monot

            t1 = t2
            ft1 = ft2
            t2 = t3
            ft2 = ft3
            t3 = t2 + step
            ft3 = self._evaluation(func_num, t3)
            monot = np.sign(ft3 - ft2)

        return endPoint, startPoint, monot

    def _opt_gold(self, func_num, a, b, eps):
        """
        黄金分割法确定最优解

        输入：
        func_num: 函数编号
        a: 左端点值
        b: 右端点值
        eps: 精度

        输出：
        result: 最优解
        """
        gc = (np.sqrt(5) - 1) / 2
        gcn = 1 - gc

        a1 = a + gcn * (b - a)
        a2 = a + gc * (b - a)

        f1 = self._evaluation(func_num, a1)
        f2 = self._evaluation(func_num, a2)

        while abs(b - a) >= eps:
            if f1 < f2:
                b = a2
                a2 = a1
                f2 = f1
                a1 = a + gcn * (b - a)
                f1 = self._evaluation(func_num, a1)
            elif f1 > f2:
                a = a1
                a1 = a2
                f1 = f2
                a2 = a + gc * (b - a)
                f2 = self._evaluation(func_num, a2)
            else:
                medianVal = 0.5 * (a + b)
                a = medianVal
                b = medianVal
                break

        result = 0.5 * (a + b)
        return result

    def _gold2(self, func_num, a, b, eps):
        """
        另一种黄金分割法实现 (0.3/0.7分割)
        """
        while abs(b - a) >= eps:
            x1 = a + 0.3 * (b - a)
            x2 = a + 0.7 * (b - a)  # 黄金分割法主要步骤
            f1 = self._evaluation(func_num, x1)
            f2 = self._evaluation(func_num, x2)

            if f1 < f2:  # 两种情形的判断
                b = x2
            else:
                a = x1

        x = (a + b) / 2  # 得到满足条件的最优解
        return x


    def _evaluation_old(self, func_num, x):
        """
        评估函数（对应Matlab中的Evaluation函数）

        输入：
        func_num: 函数编号
        x: 标量值（将被放入solution的rootVar位置）

        输出：
        fitness: 函数值
        """
        # 创建完整的解向量
        solution = self.cv.copy() if self.cv is not None else np.zeros(self.info['dimension'])

        # 设置当前检测变量的值
        if self.rootVar is not None:
            solution[self.rootVar] = x

        # 如果正在进行扰动，设置扰动组的值
        if self.isPerturb and self.perturb is not None and self.detectingGroup is not None:
            solution[self.detectingGroup] = self.perturb

        # 确保在边界内
        solution = np.clip(solution, self.info['lower'], self.info['upper'])

        # 评估函数
        fitness = self.fun(solution)

        # 更新函数评估计数
        self.A_FEs += 1

        return fitness

    def evaluate_function(self, x):
        """
        统一的函数评估接口
        """
        x = np.clip(x, self.info['lower'], self.info['upper'])
        return self.fun(x)

    def interaction_detection(self, rootVar, detectingGroup, A_localOpt,
                              detectionStep, alpha, func_num, eps, opts):
        """
        交互检测算法

        输入：
        rootVar: 当前检测的变量索引
        detectingGroup: 当前检测的变量组
        A_localOpt: 局部最优值
        detectionStep: 检测步长
        alpha: 误差阈值系数
        func_num: 函数编号
        eps: 黄金分割法精度
        opts: 选项字典

        输出：
        A__isInteract: 是否交互 (0: 不交互, 1: 交互, nan: 边界条件)
        A_localOpt_P: 扰动后的局部最优值
        detectionStep: 更新后的检测步长
        A_deltaLeft: 左侧差分
        A_deltaRight: 右侧差分
        """
        # 初始化输出
        A_localOpt_P = np.nan
        A_deltaLeft = np.nan
        A_deltaRight = np.nan

        # 获取边界
        lb = self.info['lower']
        ub = self.info['upper']

        # 创建扰动点
        if self.cv is None:
            self.cv = np.zeros(self.info['dimension'])

        # 创建三个扰动点副本
        LOL_Perturbed = self.cv.copy()
        LO_Perturbed = self.cv.copy()
        LOR_Perturbed = self.cv.copy()

        # 设置扰动组的值
        if self.perturb is not None:
            LOL_Perturbed[detectingGroup] = self.perturb
            LO_Perturbed[detectingGroup] = self.perturb
            LOR_Perturbed[detectingGroup] = self.perturb

        # 设置当前变量的最优值
        LO_Perturbed[rootVar] = A_localOpt

        # 计算中心点的函数值
        fitO_Perturbed = self._evaluation(func_num, A_localOpt,
                                          rootVar, detectingGroup, LO_Perturbed)

        # 计算误差阈值（简化版本）
        errorThreshold = max(1e-6, abs(fitO_Perturbed) * alpha)

        # 初始化差分值
        A_deltaLeft = fitO_Perturbed * 0  # 设为0，确保进入循环
        A_deltaRight = fitO_Perturbed * 0

        # 标志位
        oneSideZero = 0

        # 主要检测循环
        while abs(A_deltaLeft) <= errorThreshold or abs(A_deltaRight) <= errorThreshold:
            # 检查单边零的情况
            if abs(A_deltaLeft) <= errorThreshold and abs(A_deltaRight) > errorThreshold:
                oneSideZero = 1
            elif abs(A_deltaLeft) > errorThreshold and abs(A_deltaRight) <= errorThreshold:
                oneSideZero = 1

            # 检查边界条件
            if A_localOpt - detectionStep < lb or A_localOpt + detectionStep > ub:
                break

            # 设置左右检测点
            left_point = A_localOpt - detectionStep
            right_point = A_localOpt + detectionStep

            # 计算左右点的函数值
            fitL_Perturbed = self._evaluation(func_num, left_point,
                                              rootVar, detectingGroup, LOL_Perturbed)
            fitR_Perturbed = self._evaluation(func_num, right_point,
                                              rootVar, detectingGroup, LOR_Perturbed)

            # 计算差分
            A_deltaLeft = fitL_Perturbed - fitO_Perturbed
            A_deltaRight = fitR_Perturbed - fitO_Perturbed

            # 增大检测步长
            detectionStep = detectionStep * 10

            # 如果步长过大，则退出（简化版本）
            if detectionStep > ub * 0.2:
                break

        # 判断交互关系
        A__isInteract = 0  # 默认可分

        # 检查边界条件
        if A_localOpt - detectionStep < lb or A_localOpt + detectionStep > ub:
            if self.outBoundFlag == 0:
                self.outBoundFlag = 1
                return np.nan, A_localOpt_P, detectionStep, A_deltaLeft, A_deltaRight

            self.outBoundTimes += 1
            isPerturb_backup = self.isPerturb

            # 使用进退法/黄金分割法重新计算局部最优
            self.isPerturb = True
            opts_bound = {'lbound': lb, 'ubound': ub}
            A_localOpt_P, _ = self.advance_retreat_gold(func_num, A_localOpt,
                                                        detectionStep / 10, eps, opts_bound)
            self.isPerturb = isPerturb_backup

            if abs(A_localOpt - A_localOpt_P) <= detectionStep:
                A__isInteract = 0
            else:
                A__isInteract = 1
        else:
            # 标准判断逻辑
            if A_deltaLeft >= errorThreshold and A_deltaRight >= errorThreshold:
                A__isInteract = 0  # 不交互
            else:
                A__isInteract = 1  # 交互

        # 重置边界标志
        self.outBoundFlag = 0

        # 更新最大检测步长统计
        if detectionStep > self.MaxDetectionStep:
            self.MaxDetectionStep = detectionStep

        if detectionStep > 1:
            self.MaxDeSTimes += 1

        return A__isInteract, A_localOpt_P, detectionStep, A_deltaLeft, A_deltaRight

    def _evaluation(self, func_num, x, rootVar=None, detectingGroup=None, solution=None):
        """
        增强版评估函数，支持指定变量和扰动组

        输入：
        func_num: 函数编号
        x: 标量值（将被放入solution的rootVar位置）
        rootVar: 变量索引（可选，覆盖self.rootVar）
        detectingGroup: 变量组（可选，覆盖self.detectingGroup）
        solution: 基础解向量（可选，覆盖self.cv）

        输出：
        fitness: 函数值
        """
        # 创建或复制解向量
        if solution is None:
            solution = self.cv.copy() if self.cv is not None else np.zeros(self.info['dimension'])
        else:
            solution = solution.copy()

        # 设置当前检测变量的值
        if rootVar is not None:
            solution[rootVar] = x
        elif self.rootVar is not None:
            solution[self.rootVar] = x

        # 如果正在进行扰动，设置扰动组的值
        if detectingGroup is not None:
            if self.perturb is not None:
                solution[detectingGroup] = self.perturb
        elif self.isPerturb and self.perturb is not None and self.detectingGroup is not None:
            solution[self.detectingGroup] = self.perturb

        # 确保在边界内
        solution = np.clip(solution, self.info['lower'], self.info['upper'])

        # 评估函数
        fitness = self.fun(solution)

        # 更新函数评估计数
        self.A_FEs += 1

        return fitness






























