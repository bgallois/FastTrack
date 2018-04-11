import numpy as np
import scipy.stats
import pandas as pd
import glob
import os
import cv2
import shutil
#import beautifulplot as bp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from bokeh.plotting import figure, show, output_file, save
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.models import LogColorMapper, LogTicker, ColorBar
from bokeh.io import export_png
from bokeh.io import export_svgs
import seaborn as sns
from detect_peaks import detect_peaks
import beautifulplot as bp
import sys
import traceback
plt.style.use('seaborn-colorblind')



class Trajectory:

    """A simple class to analyse output of Fishy software"""



    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path + '/tracking.txt', sep="  ", engine='python',na_values=[' nan'])
        self.Milestones = pd.read_csv(path + '/Milestones.txt', sep='\t', engine='python', header=None)
        #self.data = self.data.dropna() # Only with one fish
        for n, time in zip(self.Milestones[0], self.Milestones[1]):
            self.data[' imageNumber'][n] = time
        self.data[' imageNumber']=  self.data[' imageNumber'].astype('float64')
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
        """
        Description: Extract the head parameters

        :fishNumber: number of the fish to extract the parameter
        :return: x position, y position, orientation in radian, timestamp
        :type return: array of doubles
        """
       
        x = self.data.iloc[self.index[fishNumber], [0]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [1]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [2]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x['xHead'].values, y[' yHead'].values, orientation[' tHead'].values, t[' imageNumber'].values




    def getTailPosition(self, fishNumber):
        """
        Description: Extract the tail parameters

        :fishNumber: number of the fish to extract the parameter
        :return: x position, y position, orientation in radian, timestamp
        :type return: array of doubles
        """
       
        x = self.data.iloc[self.index[fishNumber], [3]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [4]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [5]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x[' xTail'].values, y[' yTail'].values, orientation[' tTail'].values, t[' imageNumber'].values




    def getCenterPosition(self, fishNumber):
        """
        Description: Extract the center of mass parameters

        :fishNumber: number of the fish to extract the parameter
        :return: x position, y position, orientation in radian, timestamp
        :type return: array of doubles
        """
        x = self.data.iloc[self.index[fishNumber], [6]].reset_index(drop=True)
        y = self.data.iloc[self.index[fishNumber], [7]].reset_index(drop=True)
        orientation = self.data.iloc[self.index[fishNumber], [8]].reset_index(drop=True)
        t = self.data.iloc[self.index[fishNumber], [10]].reset_index(drop=True)

        return x[' xBody'].values, y[' yBody'].values, orientation[' tBody'].values, t[' imageNumber'].values




    def getCurvature(self, fishNumber):
        """
        Description: Extract the curvature of the fish

        :fishNumber: number of the fish to extract the parameter
        :return: curvature
        :type return: array of doubles
        """

        curv = self.data.iloc[self.index[fishNumber], [9]].reset_index(drop=True)
       
        return curv[' curvature'].values


    def getDisplacement(self, fishNumber):

        displacement = np.sqrt((((self.data.iloc[self.shiftIndex[fishNumber], [0]].reset_index(drop = True)) - (self.data.iloc[self.index[fishNumber], [0]].reset_index(drop = True)))**2).values + (((self.data.iloc[self.shiftIndex[fishNumber], [1]].reset_index(drop = True)) - (self.data.iloc[self.index[fishNumber], [1]].reset_index(drop = True)))**2).values)
        return displacement




    def getConcentration(self, fishNumber):

        concentration = self.data.iloc[self.index[fishNumber], [11]].reset_index(drop=True)
        tmp = np.isnan(concentration[' concentration'])
        for index, conc in enumerate(tmp):
            if conc == True:
                count = 1
                while(index + count < len(tmp)-1 and tmp[index + count] == True):
                    count += 1

                if index + count < len(tmp)-1:
                    concentration[' concentration'][index] = (concentration[' concentration'][index - 1] + concentration[' concentration'][index + count])*.5
                else: 
                    concentration[' concentration'][index] = concentration[' concentration'][index - 1]

        print("done")
        concentration -= 128
        concentration = self.normalization(concentration)
       
        return concentration[' concentration'].values




    def preferenceIndex(self, fishNumber):
        '''Infamous method, have to be cleared'''

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


        act = self.getDisplacement(fishNumber)
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
        """
        Description: Plot the a graph with the x position of the head, the time and the concentration around the head by the color of the point?

        :fishNumber: number of the fish to extract the parameter
        :return: figure, save also figure un html, png, svg in a folder
        :type return: bokeh figure
        """

        x, __, __, t = self.getHeadPosition(fishNumber)
        c = self.getConcentration(fishNumber)
        refTime = t[0]
        t -= t[0]
        t = t* (1e-9/60)

        
        TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
        p = figure(tools=TOOLS, x_axis_label = "x position", y_axis_label = "Time (min)")
        colorMapper = LinearColorMapper(palette='Plasma256', low=0, high=1)
        colorBar = ColorBar(color_mapper=colorMapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0), title="Concentration")
        p.add_layout(colorBar, 'right')
        source = ColumnDataSource(dict(x=x, y=t, z=c.values))
        p.circle(x='x', y='y', fill_color={'field': 'z', 'transform': colorMapper}, line_color=None, source=source)

        for i in self.Milestones[1][:]:
            p.line([0, 1000], [(i - refTime)*1e-9/60, (i - refTime)*1e-9/60], line_width=2, line_color='red')


        concentrationE = self.path.find('pc/')
        concentrationB = self.path.find('AcideCitrique/') + len('AcideCitrique/')

        output_file('plot.html')
        #show(p)
    

        if os.path.exists('/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE]):
            save(p, '/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE] + '/' + self.path[concentrationE + 3 : concentrationE + 13] + '_' + self.path[-8::] + '_curve.html')
            export_png(p, filename='/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE] + '/' + self.path[concentrationE + 3 : concentrationE + 13] + '_' + self.path[-8::] + '_curve.png')
            p.output_backend = "svg"
            export_svgs(p, filename='/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE] + '/' + self.path[concentrationE + 3 : concentrationE + 13] + '_' + self.path[-8::] + '_curve.svg')     

        else :
            os.mkdir('/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE])
            save(p, '/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE] + '/' + self.path[concentrationE + 3 : concentrationE + 13] + '_' + self.path[-8::] + '_curve.html')
            export_png(p, filename='/home/ljp/resultatsAcideCitrique/' + self.path[concentrationB : concentrationE] + '/' + self.path[concentrationE + 3 : concentrationE + 13] + '_' + self.path[-8::] + '_curve.png')

        return p




    def normalization(self, array):
        """
        Description: Normalizarion function with a saturation for negative values.

        :array: array of doubles to normalize
        :return: normalized array
        :type return: array of doubles
        """
        array *= (array < 0)
        array = abs(1-(array - np.min(array))/(np.max(array) - np.min(array)))
        
        return array




    def extractBoutsbyCurv(self, fishNumber):
        """
        Description: Extract the swim bouts of the fish.

        :fishNumber: number of the fish to extract the parameter
        :return: swim bouts
        :type return: array of doubles
        """
        x, y, o, t = self.getHeadPosition(fishNumber)
        t = t* (1e-9)
        v = self.getDisplacement(fishNumber)[0:-1,0]
        v = v / np.diff(t)
        peakind = detect_peaks(v, mph=100, mpd=0.5)

        return peakind


    def timeInsideProduct(self, fishNumber):

        x, y, o, t = self.getHeadPosition(fishNumber)
        t -= t[0]
        t = t* (1e-9)
        c = self.getConcentration(fishNumber)
        cThresh = c * (c > .35)

        for index, conc in enumerate(cThresh):
            if (index < self.Milestones[0][3]) and (x[index] < 250 or x[index] > 750) and conc != 0:
                cThresh[index] = 0

            elif (index < self.Milestones[0][5]) and index > self.Milestones[0][3] and x[index] > 750 and conc != 0:
                cThresh[index] = 0
            elif (index < self.Milestones[0][5]) and index > self.Milestones[0][3] and x[index] < 250 and conc == 0:
                cThresh[index] = cThresh[index - 1]

            elif (index < self.Milestones[0][7]) and index > self.Milestones[0][5] and (x[index] < 250 or x[index] > 750) and conc != 0:
                cThresh[index] = 0

            elif (index < self.Milestones[0][7]) and x[index] > 750 and conc == 0:
                cThresh[index] = 0
            elif (index < self.Milestones[0][7]) and x[index] < 250 and conc != 0:
                cThresh[index] = cThresh[index - 1]



        fig, ax1 = plt.subplots()
        timeInside = 0
        deltatime = []
        it = 0

        while(it < len(c)-1):
            
            if cThresh[it] != 0.:
                count = 1
                
                while(cThresh[it + count] > 0.):
                    count += 1

                    if(it + count > len(cThresh)-1):
                        count -= 1
                        break

                
                timeInside += t[it + count] - t[it]
                deltatime.append(t[it + count] - t[it])
                ax1.add_patch(patches.Rectangle((t[it], 0), t[it + count] - t[it], 1000, alpha=0.3, color='y'))
                it += count
                

            else:
                it += 1
                
        ax1.plot(t, x, '.-')
        ax1.plot(t, y, '.-')
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('x (px)')
        ax1.axhline(500, 0, len(t), color='k')

        ax2 = ax1.twinx()
        ax2.plot(t, c, 'r.-')
        ax2.set_ylabel('Concentration', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()

        

        return (timeInside/(t[-1]-t[0]))*100, deltatime





        


        


    
    '''def caracteristicLength(self, FishNumber):

        x, __, o, t = self.getHeadPosition(fishNumber)
        c = self.getConcentration(fishNumber)
        refTime = t[0]
        t -= t[0]
        t = t* (1e-9/60)boots

        

        for index, angle in enumerate(o):


        c = abs(1-(c - np.min(c))/(np.max(c) - np.min(c))) # Normalisation

        for i in self.Milestones[1][:]:'''
            



    




   
       

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
        v = A.getDisplacement(0)
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








label = ['0.01', '0.02', '0.03', '0.06', '0.1']
folder = []
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.01pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.02pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.03pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.06pc/*/*'))
folder.append(glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.1pc/*/*'))

pool = []
dist = []

for i in folder:
    tmp = []
    tmpMat = []
    for j in i:
        try:
            B = Trajectory(j)
            d, c = B.timeInsideProduct(0)
            tmp.append(d)
            for cc in c:
                tmpMat.append(cc)


        except BaseException as e:
            print(j)
            pass

    pool.append(tmp)
    dist.append(tmpMat)

    

fig = bp.BoxPlot()
fig.plot(pool, label = label)
fig.addN()
fig.limits(ylim = [0, 50])
fig.plotPoints()
fig.addLabels(xlabel = 'Concentration g/L', ylabel = "Percentage of time in acid")

fig2 = bp.BoxPlot()
fig2.plot(dist, label = label)
fig2.addN()
fig2.plotPoints()
fig2.addLabels(xlabel = 'Concentration g/L', ylabel = "Percentage of time in acid")



'''fig3 = bp.BoxPlot()
fig3.plot(pool, label = label)'''
'''fig3.addN(size = 12)
fig3.limits(ylim = [-1, 1])
fig3.plotPoints()
fig3.addLabels(xlabel = 'Concentration g/L', ylabel = "Normalized preference index")

bp.show()'''

'''B = Trajectory('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.1pc/2018-03-06/Run 1.01/')
B.concentrationPlot(0)

x, _, _, t = B.getHeadPosition(0)
c = B.getConcentration(0)
print(np.argmax(c), np.max(c))'''

'''folder = glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/*/*/*')

for i in folder:
    try:
        Trajectory(i).concentrationPlot(0)
    except:
        print(i)
        pass'''



'''a = Trajectory('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.01pc/2018-03-07/Run 4.03')
x, y, o, t = a.getTailPosition(0)
xh, yh, oh, th = a.getHeadPosition(0)
t -= t[0]
o = np.unwrap(o) #To check
t = t* (1e-9)
c = a.getConcentration(0)
a.concentrationPlot(0)
r = a.getDisplacement(0)
c *= (c > 0.15)
changementOrient = np.diff(o)
changementOrient = (changementOrient + 180) % 360 - 180
curv = a.getCurvature(0)
bo = a.extractBoutsbyCurv(0)
l = a.getDisplacement(0)
d = a.timeInsideProduct(0)'''



'''inside = []
outside = []

for i, j in enumerate(c):
    for k in bo:
        if j != 0 and i == k:
            outside.append()
        if j == 0 and i == k:
            inside.append()

sns.distplot(outside, label='Buffer');
sns.distplot(inside, label='Acide');
plt.legend()'''

'''fig, ax1 = plt.subplots()

ax1.plot(t[a.Milestones[0][1]:a.Milestones[0][2]], x[a.Milestones[0][1]:a.Milestones[0][2]], 'b.-')
ax1.set_xlabel('time (min)')
ax1.set_ylabel('x (px)', color='b')
ax1.tick_params('y', colors='b')
ax1.plot(t[a.Milestones[0][1]:a.Milestones[0][2]], o[a.Milestones[0][1]:a.Milestones[0][2]]*(1000/6.), 'g.-')

ax2 = ax1.twinx()
ax2.plot(t[a.Milestones[0][1]:a.Milestones[0][2]], c[a.Milestones[0][1]:a.Milestones[0][2]], 'r.-')
ax2.set_ylabel('Concentration', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()'''



'''fig, ax1 = plt.subplots()

ax1.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], xh[a.Milestones[0][3]:a.Milestones[0][4]], 'b.-')
#ax1.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], xh[a.Milestones[0][3]:a.Milestones[0][4]], 'r.-')
ax1.plot(t[bo], xh[bo], 'ro')

#ax1.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], y[a.Milestones[0][3]:a.Milestones[0][4]] +1000, 'b.-')
#ax1.plot(t[bo], y[bo]+1000, 'ro')

#ax1.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], curv[a.Milestones[0][3]:a.Milestones[0][4]], 'b.-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x (px)', color='b')
ax1.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], o[a.Milestones[0][3]:a.Milestones[0][4]]*(1000/6.), 'g.-')

ax2 = ax1.twinx()
#ax2.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], v[a.Milestones[0][3]:a.Milestones[0][4]], 'r.-')
ax2.plot(t[a.Milestones[0][3]:a.Milestones[0][4]], c[a.Milestones[0][3]:a.Milestones[0][4]], 'r.-')
ax2.set_ylabel('Concentration', color='r')
ax2.plot(t[bo], c[bo], 'bo')
ax2.tick_params('y', colors='r')

fig.tight_layout()

path = glob.glob('/usr/RAID/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.1pc/2018-03-06/Run 4.01/*pgm')
path.sort()
for i, j in enumerate(path):
    frame = cv2.imread(j)
    for k in bo:
        if i == k:
            cv2.circle(frame, (int(x[i]), int(y[i])), 5, (0,0,255), -1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(250) & 0xFF == ord('q'):
            break'''


plt.show()