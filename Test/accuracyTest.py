import pytest
import os
import shutil
import time
import pandas
import numpy  as np
import shutil
import warnings

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def invertId(operations, dataframe, startTime = 0):
    """Invert two objects in a dataframe in place"""
    errors = 0
    indexes = []
    in1 = []
    in2 = []
    for i in operations:
        indexes.append(dataframe[(dataframe["id"].values == i[0]) & (dataframe["imageNumber"].values >= startTime)].index)
        in1.append(i[0])
        in2.append(i[1])
    n = []
    m = []
    for i in operations:
        if i[0] not in in2:
            n.append(i[0])
        if i[1] not in in1:
            m.append(i[1])

    ## Find the best appariemment // Never find a difference in error number if skipped ...
    if len(n) > 1:
        k = []
        l = []
        for i in n:
            for j in m:
                if (i, j) in operations:
                    k.append(i)
                    l.append(j)
                else:
                    search = i
                    while(True):
                        if search == j:
                            k.append(i)
                            l.append(j)
                            break;
                        if search not in in1:
                            break;
                        for kl in operations:
                            if kl[0] == search:
                                search = kl[1]
        n = k
        m = l
                    
    errors += len(operations)
    for i, j in zip(n, m):
        operations.append((j, i))
        indexes.append(dataframe[(dataframe["id"].values == j) & (dataframe["imageNumber"].values >= startTime)].index)
   # errors -= len(n)
        
    for i, j in enumerate(indexes):    
        dataframe["id"].loc[j] = operations[i][1]
    return errors

def errorsCounter(reference, tracking):
    #start_time = time.time()
    errors = 0
    objects = 0
    ids = set()
    ids2 = set()
    anormalDetectionBehavior = 0
    # Get a sub-dataframe for each time point
    for i in set(tracking["imageNumber"].values):
        ref = reference[reference["imageNumber"].values == i]
        track = tracking[tracking["imageNumber"].values == i]
        # Make a rapid check if the sub-dataframe are exactly the same
        # If exactly the same update id list and object number and break
        # Else iterate throught the subdataframe to correct errors
        ref = ref.sort_values(by=["id"])
        ref.reset_index(drop=True, inplace=True) 
        track = track.sort_values(by=["id"])
        track.reset_index(drop=True, inplace=True)
        if ref.size == track.size:
            test = ref == track
            if test.all().all():
                ids.update(ref["id"].values)
                ids2.update(track["id"].values)
                objects += track["id"].size
            else:
                skip = []
                operations = []
                for j in ref.itertuples():
                    tmp = track[ (j.xBody == track["xBody"].values) & (j.yBody == track["yBody"].values)]
                    objects += 1
                    if not tmp.empty and j.id != tmp["id"].values[0]:
                        operations.append((tmp["id"].values[0], j.id))
                        if tmp["id"].values[0] not in ids2 and j.id not in ids:
                            errors -= 1
                            if j.imageNumber != 0: objects -= 1
                            
                ids.update(ref["id"].values)
                ids2.update(track["id"].values)
                errors += invertId(operations, tracking, i)
                if True:
                    ref = reference[reference["imageNumber"].values == i]                                                          
                    track = tracking[tracking["imageNumber"].values == i]
                    ref = ref.sort_values(by=["id"])
                    ref.reset_index(drop=True, inplace=True)
                    track = track.sort_values(by=["id"])
                    track.reset_index(drop=True, inplace=True)
                    test = ref == track
                    if not test.all().all():
                        raise NameError('MError')
        else:
            anormalDetectionBehavior += 1
            skip = []
            operations = []
            for j in ref.itertuples():
                tmp = track[ (j.xBody == track["xBody"].values) & (j.yBody == track["yBody"].values)]
                objects += 1
                if not tmp.empty and j.id != tmp["id"].values[0]:
                    operations.append((tmp["id"].values[0], j.id))
                    if tmp["id"].values[0] not in ids2 and j.id not in ids:
                        errors -= 1
                        if j.imageNumber != 0: objects -= 1
                        
            ids.update(ref["id"].values)
            ids2.update(track["id"].values)
            errors += invertId(operations, tracking, i)
        
    objects -= tracking[tracking["imageNumber"].values == 0].id.size
    ps = float(errors) / float(objects) 
    #print("--- %s seconds ---" % (time.time() - start_time))
    return (ps, errors, objects, anormalDetectionBehavior)

def tracking(path, imagePath, normDist=None, normAngle=None, maxDist=None, maxTime=None, normArea=None, normPerim=None):
# Import truth parameters
  groundParameter = pandas.read_csv(path + "images/Groundtruth/Tracking_Result/cfg.toml", header=None, engine="python", sep=" = ")
  groundParameter = groundParameter.set_index(0)

# Set detection parameter
  maxArea = str(int(groundParameter.loc["maxArea"][1]))
  minArea = str(int(groundParameter.loc["minArea"][1]))
  lightBack = str(int(groundParameter.loc["lightBack"][1]))
  thresh = str(int(groundParameter.loc["thresh"][1]))
  reg = str(int(groundParameter.loc["reg"][1]))
  spot = str(int(groundParameter.loc["reg"][1]))
  nBack = str(int(groundParameter.loc["nBack"][1]))
  regBack = str(int(groundParameter.loc["regBack"][1]))
  methBack = str(int(groundParameter.loc["methBack"][1]))
  xTop = str(int(groundParameter.loc["xTop"][1]))
  yTop = str(int(groundParameter.loc["yTop"][1]))
  xBottom = str(int(groundParameter.loc["xBottom"][1]))
  yBottom = str(int(groundParameter.loc["yBottom"][1]))
  morph = str(int(groundParameter.loc["morph"][1]))
  morphSize = str(int(groundParameter.loc["morphSize"][1]))
  morphType = str(int(groundParameter.loc["morphType"][1]))


  cmd = "../build_cli/fasttrack-cli --maxArea " + maxArea + " --minArea " + minArea + " --lightBack "+ lightBack + " --thresh "+ thresh + " --reg " + reg + " --spot "+ spot + " --nBack "+ nBack + " --regBack "+ regBack + " --methBack " + methBack+ " --xTop "+ xTop + " --yTop "+ yTop + " --xBottom " + xBottom + " --yBottom " + yBottom+ " --morph " + morph + " --morphSize " + morphSize+ " --morphType " + morphType +" --normArea " +str(normArea) + " --normPerim "+ str(normPerim) + " --normDist " + str(normDist) + " --normAngle " + str(normAngle) + " --maxDist " + str(maxDist) + " --maxTime " + str(maxTime) + " --path "+ path + imagePath + " --backPath dataSet/images/Groundtruth/Tracking_Result/background.pgm > /dev/null 2>&1" 
  out = os.system(cmd)


def test_accuracy():
  reference = pandas.read_csv("dataSet/images/Groundtruth/Tracking_Result/tracking.txt", engine="python", sep="\t", usecols=['xBody', 'yBody', 'imageNumber', 'id'])
  tracking("dataSet/", "images/frame_000000.pgm", normDist=20, normAngle=118, maxDist=195, maxTime=40, normArea=0, normPerim=0)
  trackingData = pandas.read_csv("dataSet/images/Tracking_Result/tracking.txt", engine="python", sep="\t", usecols=['xBody', 'yBody', 'imageNumber', 'id'])              
  res = errorsCounter(reference, trackingData)
  shutil.rmtree("dataSet/images/Tracking_Result/")
  assert res[1] == 11, "If better accuracy than previously reported! Please update the benchmark comment and the test with: " + str(np.around(res[0]*100, 2)) + '%\t' + str(res[1]) + " errors"
  """
  v5.0.0 accuracy: 0.49%	12 errors  
  vdev accuracy: 0.45%	11 errors by adding correlation between objets histogram intensity without normalisation  
  """

