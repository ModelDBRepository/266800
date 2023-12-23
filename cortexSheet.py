import numpy as np
import nodes
import time
import random
np.seterr(all='ignore');        #ignore overflow warnings in sigmoid calculations when the y is small
         
class cortexSheet():
    def __init__(self, xsize, ysize, v1v2b, sigma_correlation, sigma_propagation, xNodeSpacing, yNodeSpacing):
        self.xsize = xsize;
        self.ysize = ysize;
        self.xNodeSpacing = xNodeSpacing; 
        self.yNodeSpacing = yNodeSpacing; 
        self.v1v2b = v1v2b; 
        self.anchors = []    #cortical coordinates of anchor nodes
        self.nonAnchors = []; 
        self.nodes = {};        #keys are cortical coordinates, values are node objects
        self.timestamp = 0;
        self.nEdgesTotal = 0;
        self.nEdgesAvg = 0; 
        self.distancePairs = {}; 
        self.scaleFactor = 1.0; 
        self.grid2node = {};    #keys are grid location, values are (randomized) node locations within the grid

        self.correlation_spread = [2*sigma_correlation[0]**2, 2*sigma_correlation[1]**2]; 
        self.propagation_spread = [2*sigma_propagation[0]**2, 2*sigma_propagation[1]**2]; 
 
        self.foutDecay = 0.1; 
        self.finDecay = 0.05; 
    
        self.cMatrix = 0;       #Matrix storing correlations of each anchor to every anchor  
        self.wMatrix = 0;       #Matrix storing number of edges from each anchor to every non-anchor
        self.dMatrix = 0;       #Matrix storing distance-effect of each non-anchor to every non-anchor
        self.aMatrix = 0; 

        self.finVector = 0; 
        self.foutVector = 0; 
        self.rMatrix = 0;       #Matrix storing resource factors
        
        random.seed(1); 
        np.random.seed(8); 
 
    def addNode(self, loc, anchor=0):
        #loc = location = cortical coordinates
        linID = len(self.anchors) + len(self.nonAnchors); 
        randomizedLocX = random.randint(-self.xNodeSpacing/2, self.xNodeSpacing/2); 
        randomizedLocY = random.randint(-self.yNodeSpacing/2, self.yNodeSpacing/2);
        randomizedLoc = (loc[0]+randomizedLocX, loc[1]+randomizedLocY); 
        loc1 = randomizedLoc; 
        #loc1 = loc; 
        self.nodes[loc1] = nodes.node(loc1, linID);
        self.grid2node[loc] = loc1; 
        if anchor == 1:
            self.anchors.append(loc1);
            self.nodes[loc1].anchor = 1;
            self.nodes[loc1].nEdgesIn = 10000000;
        else:
            self.nonAnchors.append(loc1);

    def addEdge(self, node_src, node_dest, wd = 1, init=0):
        #wd is # of edges between src and dest
        if node_src in self.nodes.keys() and node_dest in self.nodes.keys():
            self.nodes[node_src].makeOutwardConnection(node_dest, wd, init); 
            self.nodes[node_dest].makeInwardConnection(node_src, wd, init);             
            self.nEdgesTotal += wd;

    def addMultipleNodes(self, xloc, yloc, anchor=0):
        xstart = xloc[0]
        xend = xloc[1]
        ystart = yloc[0]
        yend = yloc[1]
        for i in range(xstart, xend, self.xNodeSpacing):
            for j in range(ystart, yend, self.yNodeSpacing):
                self.addNode((i,j), anchor=anchor)
                                                                   
    def getColors(self):
        self.updateColors(); 
        out = {}; 
        for i in self.nodes.keys():
            out[i] = (self.nodes[i].color, self.nodes[i].colorIntensity);
        return out; 

    def getVisualAngles(self):
        visualAngles = {}; 
        for i in self.nodes.keys():
            visualAngles[i] = self.nodes[i].visualAngle
        return visualAngles; 
 
    def initAnchorColorMap(self): 
        nanchorsY = self.v1v2b/self.yNodeSpacing; 
        nanchorsX = self.xsize/self.xNodeSpacing; 
        incY = 1.0/float(nanchorsY);
        incX = 1.0/float(self.xsize/self.xNodeSpacing); 
        visualAngleIncrementY = 1.0/(nanchorsY);
        visualAngleIncrementX = 1.0/(nanchorsX); 
        visualAngleX = visualAngleIncrementX;
        for x in range(0, self.xsize/self.xNodeSpacing): 
            #visualAngleY = 0.1;
            visualAngleY = visualAngleIncrementY;
            for y in range(0, nanchorsY):
                red = round(1.0 - incY/2.0 - y*incY, 3);
                green = np.clip(round(x*incX - y*incY, 3), 0, 1);    
                #green = round(incX/2.0 + x*incX, 3);
                #green = 0; 
                blue = round(incX/2.0 + x*incX, 3);
                gridLoc = (x*self.xNodeSpacing, y*self.yNodeSpacing);
                node = self.grid2node[gridLoc]; 
                self.nodes[node].color = (red, green, blue); 
                self.nodes[node].colorIntensity = 1.0;
                self.nodes[node].visualAngle = (visualAngleX, visualAngleY); 
                visualAngleY = round(visualAngleY + visualAngleIncrementY, 4);
            visualAngleX = round(visualAngleX + visualAngleIncrementX, 4);
 
    def updateColors(self):
        for x in range(0, self.xsize, self.xNodeSpacing):
            for y in range(0, self.ysize, self.yNodeSpacing):
                #i=(x,y); 
                i = self.grid2node[(x, y)];
                if self.nodes[i].anchor==0:
                #if self.nodes[i].anchor!=2:
                    weightSum = 0;
                    r, g, b, alpha = 0, 0, 0, 0;
                    visualAngleX, visualAngleSumX = 0, 0;
                    visualAngleY, visualAngleSumY = 0, 0; 
                    for j in self.nodes[i].rfs.keys():
                        #if j in nodes.keys():
                        if self.nodes[j].anchor==1:
                                w = self.nodes[i].rfs[j]; 
                                r = r + w*self.nodes[j].color[0]; 
                                g = g + w*self.nodes[j].color[1]; 
                                b = b + w*self.nodes[j].color[2];
                                alpha = alpha + w*self.nodes[j].colorIntensity;  
                                weightSum = weightSum + w;
                                visualAngleSumX += w*self.nodes[j].visualAngle[0];
                                visualAngleSumY += w*self.nodes[j].visualAngle[1]; 	         
                    if (weightSum!=0): 
                        #Averaging
                        r = round(r/float(weightSum), 4); 
                        g = round(g/float(weightSum), 4);  
                        b = round(b/float(weightSum), 4);
                        alpha = round(alpha/float(weightSum), 4);
                        visualAngleX = round(visualAngleSumX/float(weightSum), 4);
                        visualAngleY = round(visualAngleSumY/float(weightSum), 4);
                    color = (r, g, b);
                    self.nodes[i].color = color;
                    self.nodes[i].colorIntensity = alpha;
                    self.nodes[i].visualAngle = (visualAngleX, visualAngleY); 

         
    def findCorrelation(self, node1, node2):
        xdiff = self.nodes[node1].loc[0] - self.nodes[node2].loc[0]; 
        ydiff = self.nodes[node1].loc[1] - self.nodes[node2].loc[1]; 

        xterm = (xdiff**2)/float(self.correlation_spread[0]);
        yterm = (ydiff**2)/float(self.correlation_spread[1]);
        c = self.scaleFactor*np.exp(-xterm-yterm); 
        return c; 

    def findCorrelationArray(self, locSeed, xLocs, yLocs):
        xterm = ((locSeed[0]-xLocs)**2)/float(self.correlation_spread[0]); 
        yterm = ((locSeed[1]-yLocs)**2)/float(self.correlation_spread[1]);
        p = np.exp(-xterm-yterm); 

        #Set values below 0.1 to 0
        indices = p<0.01;
        #indices = p<0.1;
        p[indices] = 0; 
        return p;  

    def findPropagation(self, node1, node2):
        xdiff = (node1[0]-node2[0]); 
        ydiff = (node1[1]-node2[1]);
        xterm = (xdiff**2)/float(self.propagation_spread[0]);
        yterm = (ydiff**2)/float(self.propagation_spread[1]);
        p = self.scaleFactor*np.exp(-xterm-yterm); 
        return p; 

    def findPropagationArray(self, locSeed, xLocs, yLocs): 
        xterm = ((locSeed[0]-xLocs)**2)/float(self.propagation_spread[0]); 
        yterm = ((locSeed[1]-yLocs)**2)/float(self.propagation_spread[1]);
        p = np.exp(-xterm-yterm); 

        #Set values below 0.1 to 0
        indices = p<0.01; 
        p[indices] = 0; 

        return p;  
              
    def findAmatrix(self):
        x = np.matmul(self.wMatrix, self.dMatrix);
        self.aMatrix = np.matmul(self.cMatrix, x); 
                                    
    def sigmoid(self, x, A=0.1, B=-0.5, C=0, D=1):
        denom = 1 + A*np.exp(-B*(x+C));
        y = float(D)/denom;
        round(y, 2); 
        return y
        
    def findFout(self, node): 
        Nout = self.nodes[node].nEdgesOut - self.nEdgesAvg;    
        fout = self.sigmoid(Nout, A=0.1, B=-self.foutDecay, D=1.0);
        if fout<0.1: fout=0; 
        return fout; 

    def findFin(self, node): 
        Nin = self.nodes[node].nEdgesIn;           
        fin = self.sigmoid(Nin, A=0.1, B=-self.finDecay, D=1.0);       #B is the slope
        if fin<0.1: fin=0; 
        return fin; 

    def findXYvectors(self, nodeLocs): 
        n = len(nodeLocs); 
        xLocs = np.zeros(n, dtype=np.float32); 
        yLocs = np.zeros(n, dtype=np.float32);
        for i in range(0, n): 
            xLocs[i] = nodeLocs[i][0]; 
            yLocs[i] = nodeLocs[i][1];  
        return xLocs, yLocs; 
 
    def findCmatrix(self):
        n = len(self.anchors); 
        self.cMatrix = np.zeros((n, n), dtype=np.float32);
        anchorXlocs, anchorYlocs = self.findXYvectors(self.anchors); 
        for i in range(0, n):
            locSeed = np.asarray(self.anchors[i]); 
            self.cMatrix[i] = self.findCorrelationArray(locSeed, anchorXlocs, anchorYlocs); 
 
    def initWmatrix(self): 
        n = len(self.anchors); 
        m = len(self.nonAnchors);
        self.wMatrix = np.zeros((n, n+m), dtype=np.float32);
        #Adding self connections in the anchors
        np.fill_diagonal(self.wMatrix, 1); 
                            
    def findDmatrix(self): 
        m = len(self.anchors); 
        n = len(self.nonAnchors);
        self.dMatrix = np.zeros((m+n, n), dtype=np.float32);
        nonAnchorXlocs, nonAnchorYlocs = self.findXYvectors(self.nonAnchors); 
        #Anchors to non-anchors
        for i in range(0, m):
            locSeed = np.asarray(self.anchors[i]);
            self.dMatrix[i] = self.findPropagationArray(locSeed, nonAnchorXlocs, nonAnchorYlocs); 
        #Non-anchors to non-anchors
        for i in range(0, n):
            locSeed = np.asarray(self.nonAnchors[i]);
            self.dMatrix[m+i] = self.findPropagationArray(locSeed, nonAnchorXlocs, nonAnchorYlocs); 
        #self.dMatrix = sparse.csr_matrix(self.dMatrix); 

    def findRmatrix(self):
        m = len(self.nonAnchors);
        self.finVector = np.zeros((m, 1), dtype=np.float32); 
        for i in range(0, m): self.finVector[i] = self.findFin(self.nonAnchors[i]); 

        n = len(self.anchors);
        self.foutVector = np.zeros((n, 1), dtype=np.float32);                                  
        for i in range(0, n): self.foutVector[i] = self.findFout(self.anchors[i]); 

        self.rMatrix = self.foutVector*self.finVector.T; 
        
    def updateRmatrix(self, newSrc, newDst): 
        self.finVector[newDst] = self.findFin(self.nonAnchors[newDst]); 
        self.foutVector[newSrc] = self.findFout(self.anchors[newSrc]); 
        self.rMatrix[newSrc, :] = self.foutVector[newSrc][0]*self.finVector.T;
        self.rMatrix[:, newDst] = self.finVector[newDst][0]*self.foutVector.T; 
                              
    def updateAmatrix(self, newSrc, newDst, wd=1):
        cVector = self.cMatrix[newSrc, :].reshape((len(self.anchors), 1));
        dVector = self.dMatrix[newDst, :].reshape((1, len(self.nonAnchors)));
        
        incMatrix = wd*cVector*dVector;
        self.aMatrix = self.aMatrix + incMatrix;
         
    def updateFactors(self, newSrc, newDst, wd=1):
        self.updateRmatrix(newSrc, newDst); 

    #Find probability distribution to connect new edge
    def updatePmatrix(self):
        p = np.multiply(self.aMatrix, self.rMatrix);
        sum1 = float(p.sum());
        if(sum1!=0):
            self.pMatrix = p/sum1; 
            return 1; 
        else:
            return 0;
       
    def sampleFromPDF(self, n=1):
        newPairs = []; 
        pdf = np.ndarray.flatten(self.pMatrix); 
        pdfIndices = np.arange(pdf.size);                 
        randomIndices = np.random.choice(pdfIndices, size=n, p=pdf);
        nAnchors = len(self.anchors); 
        for i in randomIndices: 
            newSrc = i/len(self.pMatrix[0]); 
            newDst = i%len(self.pMatrix[0]);
            newPairs.append((newSrc, newDst));
            #self.wMatrix[newSrc, newDst] = self.wMatrix[newSrc, newDst] + 1;  
            self.wMatrix[newSrc, nAnchors+newDst] = self.wMatrix[newSrc, nAnchors+newDst] + 1;  
        return newPairs; 

    #Find destination of new edge based on probability distribution defined by self.pMatrix
    def findNewPairs(self, n=1): 
        flag = self.updatePmatrix();
        #flag = self.updatePmatrixRandomized();
        newPairs = {};                           
        if flag==1: 
            #for i in range(0, n):
            pairs = self.sampleFromPDF(n);
            for pair in pairs:     
                if pair not in newPairs.keys():
                    newPairs[pair]= 1; 
                else:
                    newPairs[pair] = newPairs[pair] + 1;
        return newPairs; 

    def initializeNetwork(self): 
        t0=time.time(); 
        self.initAnchorColorMap(); 
        self.initWmatrix(); 
        self.findDmatrix();
        self.findCmatrix();
        self.findAmatrix(); 
        self.findRmatrix(); 
        self.updatePmatrix();
        t1=time.time(); 
        print("Initialization Time ="+str(t1-t0)+"s")
        
    def growNetwork(self, nSteps=1, edgesPerStep=1, monitor=0, disp=0):
        for i in range(0, nSteps):             
            self.timestamp += 1;
            newPairs = self.findNewPairs(n=edgesPerStep);
            if len(newPairs) == 0: 
                return 0; 
            for i in newPairs.keys(): 
                srcNode = i[0]; 
                dstNode = i[1]; 
                wd = newPairs[i]; 
                self.addEdge(self.anchors[srcNode], self.nonAnchors[dstNode], wd=wd, init=0);
                self.updateFactors(srcNode, dstNode, wd=wd);            
            self.findAmatrix();  

#            return 1; 

