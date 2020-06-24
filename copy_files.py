import os
import shutil

src_dir = r'c:\tmp\pics\201909'
dst_dir = r'C:\tmp\output\JPEGImages'

def get_file_name(in_file):
    name = in_file[:-4]
    out_file = '%s.jpg'%name
    return out_file

alllist=os.listdir(r'C:\tmp\output\Annotation')
for in_file in alllist:
    file_name = get_file_name(in_file)
    oldname = os.path.join(src_dir,file_name)
    newname = os.path.join(dst_dir,file_name)
    shutil.copyfile(oldname,newname)