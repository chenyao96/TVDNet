import os
import shutil

def allocation():
    dir = './data/test/'
    files = os.listdir(dir)
    i=0
    for file in files:
        i += 1
        if(i%10==0):
            jpg_from = './data/train/' +file
            jpg_to = './data/val/' + file
            mat_from = './labels/train/' + file +'.mat'
            mat_to = './labels/val/' + file + '.mat'
            print(jpg_from, " -> ", jpg_to)
            print(mat_from, " -> ", mat_to)
            shutil.move(jpg_from, jpg_to)
            shutil.move(mat_from, mat_to)


if __name__ == '__main__':
    allocation()
    pass