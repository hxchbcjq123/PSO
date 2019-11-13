import random

from particle import PT
from fitnessfunction import fitness


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
        particle = PT(psotest.initposition(self.Xdmax), psotest.initvelocity(self.Vdmax))
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
            p.v[d] = self.w * x.v[d] + self.c1 * random.random() * (pi.v[d] - x.v[d]) + self.c2 * random.random() * (
                    pg.v[d] - x.v[d])
        return p

    def updatex(self, x):  # 更新位置和适应度
        p = self.copyparticle(x)
        for d in range(self.Dim):
            p.x[d] = p.x[d] + p.v[d]
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


if __name__ == "__main__":
    psotest = PSO(fitness.Sf7, 3, 1, 1, 5, 0.8, 122, 2, 20, 100)
    p = psotest.particlelist(20)
    pi = psotest.initpilist(p)
    a = psotest.getgbest(p)
    b = psotest.getgbest(p[10:20])
    pi[1] = psotest.updatev(pi[1], a, b)
    pi[1] = psotest.updatex(pi[1])
    s = fitness.Griewank(p[2])
    s1 = fitness.Sf7(p[2])
    print(s1)
    print(p[2].x, p[2].v, p[2].f)
