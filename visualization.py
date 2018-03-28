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
        self.data = pd.read_csv(path + 'tracking.txt', sep="  ", engine='python',na_values=[' nan'])
        self.Milestones = pd.read_csv(path + 'Milestones.txt', sep='\t', engine='python', header=None)
        self.data = self.data.dropna() # Only with one fish
        self.nmax = self.objectNumber()
        self.index, self.shiftIndex = self.indexing()







    def objectNumber(self):
        """
        Description: Count the number of object tracked by Fishy software.


        :return: number of objects
        :type return: int

        """
        count = 0
        id = self.data[' imageNumber'][0]
        for i, j in enumerate(self.data[' imageNumber']):
            if j == id:
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
        refTime = t[0]
        t -= t[0]
        t = t* (1e-9/60)

        c = abs(1-(c - np.min(c))/(np.max(c) - np.min(c)))
        
        TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
        p = figure(tools=TOOLS, x_axis_label = "x position", y_axis_label = "Time (min)")
        colorMapper = LinearColorMapper(palette='Plasma256', low=np.percentile(c, 1), high=np.percentile(c, 99))
        colorBar = ColorBar(color_mapper=colorMapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0), title="Concentration")
        p.add_layout(colorBar, 'right')
        source = ColumnDataSource(dict(x=x, y=t, z=c.values))
        p.circle(x='x', y='y', fill_color={'field': 'z', 'transform': colorMapper}, line_color=None, source=source)

        for i in self.Milestones[1][:]:
            p.line([0, 1000], [(i - refTime)*1e-9/60, (i - refTime)*1e-9/60], line_width=2, line_color='black', line_dash='dotted')

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

B = Trajectory('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.01pc/2018-02-28/Run 1.02/')
B.concentrationPlot(0)

x, _, _, t = B.getHeadPosition(0)
c = B.getConcentration(0)
print(np.argmax(c), np.max(c))



