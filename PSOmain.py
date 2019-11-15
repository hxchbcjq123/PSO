from PSO import PSO
from fitnessfunction import fitness
from PSOplot import psoplt

if __name__ == "__main__":
    psotest = PSO(fitness.Tablet, 5, 5, 0.6, 1.4, 1.4, 5, 10, 20, 100)
    # 初始化方法参数分别为fitfunc, Dim, M, w, c1, c2, k1, k2, Vdmax, Xdmax
    p = psotest.particlelist(20)  # 初始化一个粒子为20的种群
    gbest = []  # 记录全局最优解
    gbest.append(psotest.getgbest(p))  # 初始化全局最好位置
    pi = psotest.initpilist(p)  # 初始化记录粒子的最好位置
    td = psotest.initTdlist(0.5)  # 初始化粒子群的速度阈值设为20
    gd = psotest.initgdlist()  # 初始化记录粒子群逃逸次数的列表
    olist = psotest.initoperator()  # 初始化算子
    for i in range(400):  # 迭代次数
        for j in range(len(pi)):  # 更新粒子最好适应度
            if p[j].f < pi[j].f:
                pi[j] = psotest.copyparticle(p[j])
        if psotest.getgbest(p).f < gbest[-1].f:
            gbest.append(psotest.getgbest(p))  # 更新全局最好速度
        for k in range(len(p)):  # 更新粒子速度
            p[k] = psotest.updatev(p[k], pi[k], gbest[-1])  # 传入参数分别为当前粒子，当前粒子最好位置，全局粒子最好位置
        p=psotest.escape(td, p, olist)  # 逃逸操作
        for l in range(len(p)):  # 更新粒子位置和适应度
            p[l] = psotest.updatex(p[l])
        Fitlist = psotest.initFitXlist(p)
        if max(Fitlist)==min(Fitlist):#适应度相同退出
            break
        olist = psotest.updateFitXlist(Fitlist, olist)  # 更新算子
        psotest.updategdtd(gd, td, p)  # 更新阈值
    for m in range(len(gbest)):
        print(gbest[m].f)
    psoplt(gbest)