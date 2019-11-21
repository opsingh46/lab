import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
d=pd.read_csv("C:\\Users\\prafu\\Desktop\\fsfsd\\heart.csv")
d1=pd.cut(d["age"],bins=[20,30,50,90])
dic2={}
#print(d1)
for i in d1:
    if(i not in dic2.keys()):
        dic2[i]={1:0,0:0}

dic=list(d1)
for i in range(len(d["target"])):
    dic2[dic[i]][d["target"][i]]+=1
#print((dic2))
    


speed = []
lifespan =[]
for i in dic2.keys():
    speed.append(dic2[i][1])
    lifespan.append(dic2[i][0])
#print(speed)

index = ['20-30', '30-50', '50-90']

df = pd.DataFrame({'Positive': speed,'negtive': lifespan}, index=index)
ax = df.plot.bar(rot=0)
plt.show()
