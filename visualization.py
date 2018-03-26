import numpy as np
import scipy.stats
import pandas as pd
import glob
#import beautifulplot as bp
import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.models import LogColorMapper, LogTicker, ColorBar
#import seaborn as sns



class Trajectory:

    """A simple class to analyse output of Fishy software"""



    def __init__(self, path):
        self.data = pd.read_csv(path + 'tracking.txt', sep="  ", engine='python')
        #self.Milestones = pd.read_csv(path + 'Milestones.txt',  header=None, sep="    ", engine='python')
        self.nmax = self.objectNumber()
        self.index, self.shiftIndex = self.indexing()






    def objectNumber(self):
        """
        Description: Count the number of object tracked by Fishy software.


        :return: number of objects
        :type return: int

        """
        count = 0
        for i, j in enumerate(self.data[' imageNumber']):
            if j == 0:
                count += 1
        return count


    def indexing(self):
        """
        Description: create list of index to map the position of each fish
       
        :return: list of index.
        :rtype: tuple of lists
        """
        index = []
        shiftIndex = []
        for i in range(self.nmax):
            index.append(range(i, len(self.data), self.nmax))
            shiftIndex.append(range(i + self.nmax, len(self.data) - self.nmax, self.nmax))
        return index, shiftIndex



    def getHeadPosition(self, fishNumber):
       
        x = self.data.iloc[self.index[fishNumber], [0]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [1]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [2]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x['xHead'], y[' yHead'], orientation[' tHead'], t[' imageNumber']


    def getTailPosition(self, fishNumber):
       
        x = self.data.iloc[self.index[fishNumber], [3]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [4]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [5]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x['xTail'], y[' yTail'], orientation[' tTail'], t[' imageNumber']


    def getCenterPosition(self, fishNumber):
       
        x = self.data.iloc[self.index[fishNumber], [6]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [7]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [8]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x['xBody'], y[' yBody'], orientation[' tBody'], t[' imageNumber']

    def getCurvature(self, fishNumber):

        curv = self.data.iloc[self.index[fishNumber], [9]].reset_index(drop=True)
       
        return curv[' curvature']


    def getVelocity(self, fishNumber):

        displacement = np.sqrt((((self.data.iloc[self.shiftIndex[fishNumber], [0]].reset_index(drop = True)) - (self.data.iloc[self.index[fishNumber], [0]].reset_index(drop = True)))**2).values + (((self.data.iloc[self.shiftIndex[fishNumber], [1]].reset_index(drop = True)) - (self.data.iloc[self.index[fishNumber], [1]].reset_index(drop = True)))**2).values)
       
        return displacement


    def getConcentration(self, fishNumber):

        concentration = self.data.iloc[self.index[fishNumber], [11]].reset_index(drop=True)
       
        return concentration[' concentration']


    def preferenceIndex(self, fishNumber):

        def preferenceIndexCalc(xPosition, side):
            pref = []
            ILeft = 0
            for i in xPosition['xHead']:
                if i < 500:
                    ILeft += 1

            if side == 'left':       
                I = (2*ILeft - len(xPosition['xHead'])) / (len(xPosition['xHead']))
            if side == 'right':       
                I = -(2*ILeft - len(xPosition['xHead'])) / (len(xPosition['xHead']))

            return I



        x, y, t = self.getHeadPosition(fishNumber)

        pref = []
        try:
            pref.append(preferenceIndexCalc(x[self.Milestones[0][1] : self.Milestones[0][2]], 'left'))
            pref.append(preferenceIndexCalc(x[self.Milestones[0][3] : self.Milestones[0][4]], 'left'))
            pref.append(preferenceIndexCalc(x[self.Milestones[0][5] : self.Milestones[0][6]], 'right'))
            pref.append(preferenceIndexCalc(x[self.Milestones[0][7] : :], 'right'))

        except:
            pass



        normalizedPref = []
        try:
            normalizedPref.append((pref[1] - pref[0])/(2.))
            normalizedPref.append((pref[3] - pref[2])/(2.))

        except:
            pass


        act = self.getVelocity(fishNumber)
        activity = []
        try:
            activity.append((np.sum(act[self.Milestones[0][1] : self.Milestones[0][2]])))
            activity.append((np.sum(act[self.Milestones[0][3] : self.Milestones[0][4]])))
            activity.append((np.sum(act[self.Milestones[0][5] : self.Milestones[0][6]])))
            activity.append((np.sum(act[self.Milestones[0][7] : -2])))
        except:
            pass

        activity /= np.max(activity)


        return pref, normalizedPref, activity


    def concentrationPlot(self, fishNumber):

        x, __, __, t = self.getHeadPosition(fishNumber)
        c = self.getConcentration(fishNumber)
        
        TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
        p = figure(tools=TOOLS, x_axis_label = "x position", y_axis_label = "Time")
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(c.values))]
        p.scatter(x, t, color=colors, fill_alpha=1, line_color=None)
        colorMapper = LogColorMapper(palette="Viridis256", low=np.percentile(c, 5), high=np.percentile(c, 95))
        colorBar = ColorBar(color_mapper=colorMapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0), title="Concentration")
        p.add_layout(colorBar, 'right')

        output_file("scatter.html")

        show(p)

        return p




   
       

'''folder = glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.06pc/*/*')
left = []
leftControl = []
right = []
rightControl = []
rightN = []
leftN = []
for i in folder:
    try:
        A = Trajectory(i + '/')
        a, nI, I = A.preferenceIndex(0)
        v = A.getVelocity(0)
        leftControl.append(I[0])
        left.append(I[1])
        rightControl.append(I[2])
        right.append(I[3])
        rightN.append(nI[1])
        leftN.append(nI[0])
    except:
        print(i)
        pass
'''

'''fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()

sns.boxplot(data = [leftControl, left, rightControl, right], ax=ax0)
ax0.set_ylim(-1.2, 1.2)
ax0.set_ylabel('Preference index')
ax0.set_xticklabels(['Buffer', 'Left cycle', 'Buffer', 'Right cycle'])
ax0.set_title('Preference index')

sns.boxplot(data = [leftN, rightN], ax=ax1)
ax1.set_ylim(-1.2, 1.2)
ax1.set_ylabel('Normalized preference index')
ax1.set_xticklabels(['Left cycle', 'Right cycle'])
ax1.set_title('Normalized preference index')'''








'''label = ['0.01', '0.02', '0.03', '0.06', '0.1']
folder = []
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.01pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.02pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.03pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.06pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.1pc/*/*'))

pool = []

for i in folder:
    tmp = []
    for j in i:
        try:
            B = Trajectory(j + '/')
            I, nI, a = B.preferenceIndex(0)
            tmp.append(nI[0])
            tmp.append(nI[1])

        except:
            pass
    pool.append(tmp)



fig3 = bp.BoxPlot()
fig3.plot(pool, label = label)
fig3.addN(size = 12)
fig3.limits(ylim = [-1, 1])
fig3.plotPoints()
fig3.addLabels(xlabel = 'Concentration g/L', ylabel = "Normalized preference index")

bp.show()'''

B = Trajectory('/run/media/benjamin/HardDisk/Behavior/DualTemporary/Run 1.02/')
B.concentrationPlot(0)

x, _, _, t = B.getHeadPosition(0)
c = B.getConcentration(0)
print(np.argmax(c), np.max(c))





