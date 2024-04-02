# coding: utf-8

import csv
import os

root = "/hdd/home/xiangzi/ImageNetLoc/ILSVRC/Data/CLS-LOC"

with open("/hdd/home/xiangzi/ImageNetLoc/LOC_val_solution.csv") as csvfile:
    spamreader = csv.reader(csvfile)
    next(spamreader, None)
    for row in spamreader:
        shortname, labels = row 
        foldername = labels.split()[0]
        imagefile = "%s/val/%s.JPEG" % (root, shortname)
        targetfolder = "%s/val_parse/%s" % (root, foldername)
        targetfile = "%s/%s.JPEG" % (targetfolder, shortname)
        if not os.path.exists(targetfolder):
            os.system("mkdir %s" % targetfolder)
        os.system("cp %s %s" % (imagefile, targetfile))
