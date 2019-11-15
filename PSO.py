import random

from particle import PT
from fitnessfunction import fitness
import math
import numpy as np


class PSO:
    def __init__(self, fitfunc, Dim, M, w, c1, c2, k1, k2, Vdmax, Xdmax):
        # fitfunc为适应度函数，Dim为维度，M为种群个数，w为惯性权重，c1,c2,k1,k2，Vdmax,Xdmax为参数，
        self.fitfunc = fitfunc
        self.Dim = Dim
        self.M = M
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.Vdmax = Vdmax
        self.Xdmax = Xdmax

    def initposition(self, scale):  # 初始化一个粒子的位置
        polist = [random.randint(-scale, scale) for i in range(self.Dim)]
        return polist

    def initvelocity(self, Vscale):  # 初始化一个粒子的速度
        vlist = [random.randint(-Vscale, Vscale) for i in range(self.Dim)]
        return vlist

    def initparticle(self):  # 初始化一个粒子
        particle = PT(self.initposition(self.Xdmax), self.initvelocity(self.Vdmax))
        return particle

    def particlelist(self, num):  # 初始化一群粒子：位置，速度，适应度
        plist = []
        for i in range(num):
            plist.append(self.initparticle())
            plist[i].f = self.getfitness(plist[i])
        plist.sort(key=self.findf)  # 按照适应度排序
        return plist

    def findf(self, x):  # 按属性排序的函数
        return x.f

    def getfitness(self, particle):  # 计算一个粒子的适应度
        return self.fitfunc(particle)

    def getgbest(self, plist):  # 根据适应度得到全局最优解的粒子
        minf = plist[0].f
        minnum = 0
        for i in range(len(plist)):
            if plist[i].f < minf:
                minf = plist[i].f
                minnum = i
        return plist[minnum]

    def copyparticle(self, x):  # 复制一个粒子
        p = self.initparticle()
        p.x = x.x.copy()
        p.v = x.v.copy()
        p.f = x.f
        return p

    def updatev(self, x, pi, pg):  # 更新速度
        p = self.copyparticle(x)
        for d in range(self.Dim):  # 更新d维速度
            p.v[d] = self.w * x.v[d] + self.c1 * random.random() * (pi.x[d] - x.x[d]) + self.c2 * random.random() * (
                    pg.x[d] - x.x[d])
        return p

    def updatex(self, x):  # 更新位置和适应度
        p = self.copyparticle(x)
        for d in range(self.Dim):
            p.x[d] = p.x[d] + p.v[d]
        while abs(p.x[d]) > self.Xdmax:  # 防止越界
            p.x[d] = self.Xdmax * random.random()

        p.f = self.getfitness(p)
        return p

    def initpilist(self, plist):  # 初始化保存粒子最好位置的一个列表
        pilist = []
        for i in range(len(plist)):
            pilist.append(self.copyparticle(plist[i]))
        return pilist

    def updatepi(self, p, pi):  # 更新粒子最好列表的一个粒子
        if p.f < pi.f:
            pi = self.copyparticle(p)
        return pi

    def initoperator(self):  # 初始化算子，初始值为优化变量取值范围
        olist = []
        for i in range(self.M):
            olist.append(2 * self.Xdmax)
        return olist

    def countFitX(self, plist):  # 计算子群适应度FitX
        xsum = 0
        for i in range(len(plist)):
            xsum = xsum + plist[i].f
        return xsum / len(plist)

    def initFitXlist(self, plist):  # 计算当前的各种群适应度
        Fitlist = []
        scale = int(len(plist) / self.M)
        for i in range(self.M):
            Fitlist.append(self.countFitX(plist[scale * i:scale * (i + 1)]))
        return Fitlist

    def updateFitXlist(self, Fitlist, olist):  # 更新变异算子,按照论文公式,Fitlist为当前计算的各种群适应度,olist为算子列表
        for i in range(self.M):
            updatef = olist[i] * math.exp((self.M * Fitlist[i] - sum(Fitlist)) / (max(Fitlist)) - min(Fitlist))
            if updatef > self.Xdmax / 4:
                updatef = abs(self.Xdmax / 4 - updatef)
            olist[i] = updatef
        return olist

    def initTdlist(self, td):  # 初始化阈值列表,为一个二元列表，阈值限制种群粒子的各维速度
        Tdlist = [[td] * self.Dim for i in range(self.M)]
        return Tdlist

    def initgdlist(self):  # gdlist用于保存第d维发生逃逸总次数
        gdlist = [[0] * self.Dim for i in range(self.M)]
        return gdlist

    def updategdtd(self, gdlist, tdlist, plist):  # 更新阈值列表
        scale = int(len(plist) / self.M)
        for i in range(self.M):  # 选择种群
            choose = plist[scale * i:scale * (i + 1)]
            for j in range(self.Dim):  # 选择维度
                for k in range(scale):  # 种群中一个粒子
                    if abs(choose[k].v[j]) < tdlist[i][j]:
                        gdlist[i][j] = gdlist[i][j] + 1
                if gdlist[i][j] > self.k1:
                    gdlist[i][j] = 0
                    tdlist[i][j] = tdlist[i][j] / self.k2

    def escape(self, tdlist, plist, olist):  # 逃逸运动
        scale = int(len(plist) / self.M)
        for i in range(self.M):  # 选择种群
            choose = plist[scale * i:scale * (i + 1)]
            for j in range(self.Dim):  # 选择维度
                for k in range(scale):  # 种群中一个粒子
                    mflist = []  # 保存根据算子计算得到的适应值f
                    if abs(choose[k].v[j]) < tdlist[i][j]:  # vid<Td则需要逃逸
                        for l in range(self.M):
                            newparticle = self.copyparticle(choose[k])
                            v1 = np.random.randn() * olist[l]
                            newparticle.x[j] = newparticle.x[j] + v1
                            mflist.append(self.getfitness(newparticle))
                            if min(mflist) == mflist[-1]:
                                vl = v1
                        minf = min(mflist)
                        newparticle1 = self.copyparticle(choose[k])
                        v2 = random.random() * self.Vdmax
                        newparticle1.x[j] = newparticle1.x[j] + v2
                        vmaxf = self.getfitness(newparticle1)
                        if minf < vmaxf:
                            choose[k].v[j] = vl
                        else:
                            choose[k].v[j] = v2
        return plist


if __name__ == "__main__":
    psotest = PSO(fitness.Sf7, 5, 5, 1, 1, 0.8, 2, 2, 20, 100)
    p = psotest.particlelist(20)
    c = psotest.initoperator()
    a = psotest.initTdlist(10)
    b = psotest.initgdlist()
    psotest.updategdtd(b, a, p)
    psotest.escape(a, p, c)
