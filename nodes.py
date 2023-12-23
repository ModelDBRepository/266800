class node():
    def __init__(self, loc, linID):
        #Receptive/Projective Field. Keys are coordinates of source, values are # of axons 
        self.rfs = {}; 
        self.pfs = {}; 
        self.nEdgesIn = 0; 
        self.nEdgesOut = 0; 
        self.anchor = 0;
        self.loc = loc; 
        self.linID = linID;         
        self.kin = 1; 
        self.kout = 1; 

        #Misc settings
        self.color = (0,0,0);
        self.colorIntensity = 0.25;  
        self.visualAngle = (0, 0);
        self.topIndex = 0;  

    def makeInwardConnection(self, source, wd=1, init=0):
        if source in self.rfs.keys():
            self.rfs[source] = self.rfs[source] + wd; 
        else:
            self.rfs[source] = wd;
                    
        if init==1:
            self.nEdgesIn += wd; 
        else:
            self.nEdgesIn += self.kin*wd;

    def makeOutwardConnection(self, dest, wd=1, init=0):
        if dest in self.pfs.keys():
            self.pfs[dest] = self.pfs[dest] + wd;
        else:
            self.pfs[dest] = wd;
                    
        if init==1: 
            self.nEdgesOut += wd; 
        else:
            self.nEdgesOut += self.kout*wd; 

