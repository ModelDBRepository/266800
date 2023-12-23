import pickle; 
import matplotlib.pyplot as plt


def plotColorsSingle(grid2node, xsize, xNodeSpacing, ysize, yNodeSpacing, colors, timestamp, ax, markerSize=50):
    for i in range(0, xsize, xNodeSpacing):
        for j in range(0, ysize, yNodeSpacing):
            nodeLoc = grid2node[(i, j)]; 
            c = colors[timestamp][nodeLoc][0];
            if c==(0, 0, 0):
                a = 0.25;
                c = (0.25, 0.25, 0.25)
            else:
                a = colors[timestamp][nodeLoc][1];
            if(c[0]<0):
                c = (0, c[1], c[2]);
            if(c[2]>1.0):
                c = (c[0], c[1], 1.0);
            
            plt.scatter(nodeLoc[0], nodeLoc[1], color=c, alpha=a, marker = 'o', s = markerSize);

    plt.xlim([-0.5,xsize-0.5]); 
    plt.ylim([-1,ysize-1]);
    plt.xticks([0, xsize]); 
    plt.yticks([0, ysize]);


def plotColorsMultiple(grid2node, xsize, xNodeSpacing, ysize, yNodeSpacing, colors1, colorLabels, timestamps = []):
    fig = plt.figure(figsize=(16,2))
    n = len(timestamps);
    tstepLabels = colorLabels;
    for i in range(0, n):
            s = str(1) + str(n) + str(i+1); 
            ax = fig.add_subplot(s); 
            plotColorsSingle(grid2node, xsize, xNodeSpacing, ysize, yNodeSpacing, colors1, timestamps[i], ax, markerSize = 2.5);
            s = 't = ' + str(tstepLabels[i]); 
            plt.title(s);
            if i==n-1:
                pass; 
            plt.xlim([-0.5,xsize-0.5]); 
            plt.ylim([0,ysize-1]); 
    plt.show()
     
rf = open('colors.pi', 'r')
colors1 = pickle.load(rf); 
colorLabels = pickle.load(rf);
grid2node = pickle.load(rf); 
xsize = pickle.load(rf); 
xNodeSpacing = pickle.load(rf); 
ysize = pickle.load(rf); 
yNodeSpacing = pickle.load(rf);


#%%
fig, ax = plt.subplots(figsize=(4,2))
#timestamp = 3;  
timestamp = 7;
markerSize = 4.0; 
plotColorsSingle(grid2node, xsize, xNodeSpacing, ysize, yNodeSpacing, colors1, timestamp, ax, markerSize = markerSize); 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


