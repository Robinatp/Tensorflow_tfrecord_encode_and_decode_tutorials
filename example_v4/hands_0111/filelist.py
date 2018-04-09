import os
import tensorflow as tf

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    print(name.split('.')[0])
                    file.write(name.split('.')[0] + "\n")
                    break

def Test():
  dir="annotations/"
  outfile="annotations/trainval.txt"
  wildcard = ".xml"  
  if os.path.exists(outfile):
      print("create please")
        
  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)
  
  file.close()

Test()