import math


class fitness:

    def Tablet(x):
        xsum = math.pow(x.x[0], 2) * math.pow(10, 6)
        xdim = len(x.x)
        for i in range(1, xdim):
            xsum += pow(x.x[i], 2)
        return xsum

    def Quadric(x):
        xsum1 = 0
        xdim = len(x.x)
        for i in range(0, xdim):
            xsum2 = 0
            for j in range(1, i + 1):
                xsum2 = xsum2 + math.pow(x.x[j], 2)
            xsum1 = xsum1 + math.pow(xsum2, 2)
        return xsum1

    def Rosenbrock(x):
        xdim = len(x.x)
        xsum = 0
        for i in range(0, xdim - 1):
            xsum = xsum + 100 * math.pow((x.x[i + 1] - math.pow(x.x[i], 2)), 2) + math.pow(x.x[i] - 1, 2)
        return xsum

    def Griewank(x):
        xdim = len(x.x)
        xsum = 0
        xmul = 1
        for i in range(xdim):
            xsum = xsum + math.pow(x.x[i], 2)
        for j in range(xdim):
            xmul = xmul * math.cos(x.x[i] / math.sqrt(i))
        return xsum / 4000 - xmul + 1

    def Rastrigin(x):
        xdim = len(x.x)
        xsum = 0
        for i in range(xdim):
            xsum = xsum + math.pow(x.x[i], 2) - 10 * math.cos(2 * math.pi * x.x[i]) + 10
        return xsum

    def Sf7(x):
        xsum = 0
        normalizer = 1.0 / float(len(x.x) - 1)
        for i in range(len(x.x) - 1):
            si = math.sqrt(x.x[i] ** 2 + x.x[i + 1] ** 2)
            xsum += (normalizer * math.sqrt(si) * (math.sin(50 * si ** 0.20) + 1)) ** 2
        return xsum