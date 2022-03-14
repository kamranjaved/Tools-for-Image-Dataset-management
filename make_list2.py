import os
import numpy as np

folder_list = ['./M']

f=open('./M.txt','w')

for dirs in folder_list:
    list_ = os.listdir(dirs)
    list_.sort()
    for i in range(len(list_)):
        print(dirs+'/'+list_[i],file=f)

f.close()
