import matplotlib.pyplot as plt

def psoplt(gblist):
    gblen=len(gblist)
    x_list=[]
    y_list=[]
    for i in range(gblen):
        x_list.append(i)
        y_list.append(gblist[i].f)
    plt.plot(x_list,y_list)
    plt.show()