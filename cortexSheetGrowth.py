import cortexSheet
import time
import numpy as np
import pickle

def init(
    ysize = 80, 
    xsize = 1,
    v1v2b = 20,
    sigma_correlation = [1.0, 1.0],
    sigma_propagation = [1.0, 1.0],
    xNodeSpacing = 1.0, 
    yNodeSpacing = 1.0 
    ):    
    cortex = cortexSheet.cortexSheet(xsize, ysize, v1v2b, sigma_correlation, sigma_propagation, xNodeSpacing=xNodeSpacing, yNodeSpacing=yNodeSpacing);
    cortex.addMultipleNodes((0, xsize), (0, v1v2b), anchor=1);        #V1 nodes
    cortex.addMultipleNodes((0, xsize), (v1v2b, ysize), anchor=0);    #Rest of the nodes
    cortex.initializeNetwork();
    np.random.seed(10);
    #np.random.seed(5);
    return cortex

#Params
print "Initializing Simulation ... " 
#simSteps = 1000;
simSteps = 350;
sigma_c = 8.0; 
sigma_p = 8.0;

sigma_correlation = [5.0, 5.0];
sigma_propagation = [40.0, 5.0];

#xsize, ysize, v1v2b, xNodeSpacing, yNodeSpacing = 500*2, 250*2, 50*2, 10, 10;      
xsize, ysize, v1v2b, xNodeSpacing, yNodeSpacing = 500*2, 250*1, 50*2, 10, 10;      

cortex = init(ysize=ysize, xsize=xsize, v1v2b=v1v2b, sigma_correlation=sigma_correlation, sigma_propagation=sigma_propagation, xNodeSpacing=xNodeSpacing, yNodeSpacing=yNodeSpacing); 
colors = [];
colorTimes = [50, 100, 150, 200, 250, 300, 350, 400, 600, 800];
c = cortex.getColors();
colors.append(c);
colorLabels = [0]; 

#%%             
t0 = time.time();
anchorPropCnt = 0; 
nAnchorPropagations = 1; 
edgesPerStep = 4*xsize/xNodeSpacing; 
 
#Main
print "Running Simulation ..."
for t in range(1, simSteps+1):
    r = cortex.growNetwork(nSteps=1, edgesPerStep=edgesPerStep, monitor=0); 
    if t in colorTimes: 
        colors.append(cortex.getColors());
        colorLabels.append(t);  
    if t%10 == 0: print t;
    if r==0:
        print("Updating nAvg"); 
        cortex.nEdgesAvg = float(cortex.nEdgesTotal)/len(cortex.anchors);
        cortex.findRmatrix();
        if cortex.rMatrix.sum()==0: 
            break;  

colors.append(cortex.getColors());
colorLabels.append(simSteps);  
t1=time.time(); 
runTime = t1-t0; 
print("Runtime="+str(runTime)+"s")

#%%                         
#Write Colors
wf = open("./colors.pi", 'w')
pickle.dump(colors, wf);
pickle.dump(colorLabels, wf);
pickle.dump(cortex.grid2node, wf); 
pickle.dump(cortex.xsize, wf);
pickle.dump(cortex.xNodeSpacing, wf); 
pickle.dump(cortex.ysize, wf);
pickle.dump(cortex.yNodeSpacing, wf);  
wf.close()
#"""

