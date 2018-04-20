import os
y='CelebA_10K' # Data folder name which is to be Rename
if os.path.isdir(y) :
   for i, filename in enumerate(os.listdir(y)):
       os.rename(y + "/" + filename, y + "/" + str(i+1) + ".jpg")
       print('Renaming image no. ', (i+1), 'out of 10000.' )


#z=os.makedirs('test_nssswssshhew2', exist_ok=True)
#for y in os.listdir("."):gfgfg
