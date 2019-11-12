import random

from particle import PT
from fitnessfunction import fitness


class PSO:
    def __init__(self, fitfunc,Dim, c1, c2, k1, k2, Vdmax, Xdmax):
        self.fitfunc=fitfunc
        self.Dim = Dim
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

    def particlelist(self,num): #初始化一群粒子
        plist=[]
        for i in range(num):
            plist.append(self.initparticle())
            plist[i].f=self.getfitness(self.fitfunc,plist[i])
        return plist

    def getfitness(self,fitfunc,particle):
        return fitfunc(particle)




if __name__ == "__main__":
    psotest = PSO(fitness.Sf7,3, 1, 1, 122, 2, 20, 100)
    p = psotest.particlelist(20)
    s=fitness.Griewank(p[2])
    s1=fitness.Sf7(p[2])
    print(s1)
    print(p[2].x,p[2].v,p[2].f)
