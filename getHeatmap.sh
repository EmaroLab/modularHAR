#!/usr/bin/env python3

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the parser
parser = argparse.ArgumentParser(description='heatmap')

# Declare an argument (`--algo`), telling that the corresponding value should be stored in the `algo` field, and using a default value if the argument isn't given
parser.add_argument('--errmatrix', action="store", dest='errMatrix', default=None)
parser.add_argument('--confmatrix', action="store", dest='confMatrix', default=None)
parser.add_argument('--width', action="store", dest='width', default=14, type = int)
parser.add_argument('--height', action="store", dest='height', default=14, type = int)
parser.add_argument('--fontscale', action="store", dest='fontScale', default=3, type = int)
parser.add_argument('--tofile', action="store", dest='toFile', default=False, type = bool)
parser.add_argument('--nullactivity', action="store", dest='nullactivity', default=False, type = bool)

# Now, parse the command line arguments and store the values in the `args` variable
args = parser.parse_args() # Individual arguments can be accessed as attributes of this object

if args.errMatrix is not None:
    filepath = args.errMatrix
    matrixDf = pd.read_csv(filepath, header = [0], index_col = [0])
    matrixDf.columns.name = 'ACTIVITY MODEL'
    matrixDf.index.name = 'ACTIVITY DATA'
    fmt = '.2g'
elif args.confMatrix is not None:
    filepath = args.confMatrix
    matrixDf = pd.read_csv(filepath, header = [0], index_col = [0])
    matrixDf.columns.name = 'ACTUAL'
    matrixDf.index.name = 'PREDICTED'
    # remove null activity from heatmap
    if not args.nullactivity:
        matrixDf = matrixDf.drop(labels='nullActivity', axis=0)
        matrixDf = matrixDf.drop(labels='nullActivity', axis=1)
    fmt='g'

sns.set(font_scale=args.fontScale)
figsize = (args.width, args.height)
fig = plt.figure(figsize=figsize)    
ax = sns.heatmap(matrixDf, annot=True, square=True, cbar = False, fmt = fmt)

if args.toFile:
    plt.savefig(f"{filepath[:-4]}.png", bbox_inches ='tight')
            #plt.close(fig) 
else:
    plt.show()

