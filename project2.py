import sys
from sklearn.svm import LinearSVC
import random as rnd;
import math;
from sklearn import svm;
from sklearn import model_selection;

###Reading Data from file
def dot_product(w1,d):
	dp1=0
	for j in range(0,cols,1):
		dp1= dp1 + (w1[j]*d[j])
	return dp1

def dotproduct (X, Y):
    dp = 0
    for j in range(0, len(Y), 1):
        dp += X[j]*float(Y[j]);
    return dp

new_data=[]
k_in=int(sys.argv[3])
datafile=sys.argv[1]
f=open(datafile)
data=[]
datax=[]
line=f.readline()
while(line !=''):
	row=line.split( )
	rowf=[]
	for i in range(0,len(row),1):
		rowf.append(float(row[i]))
	data.append(rowf)
	rowf.append(1)
	datax.append(rowf)
	line=f.readline()
num_rows=len(data)
cols=len(data[0])
f.close()

###Reading Labels from file

label_file=sys.argv[2]
f=open(label_file)
train_labels={}
line=f.readline()
num=[0,0]
while(line!=''):
	row=line.split( )
	train_labels[int(row[1])]=int(row[0])
	if int(row[0])==0:
		train_labels[int(row[1])]=-1
	line=f.readline()
	num[int(row[0])]+=1
	
'''err = open('Project_Hyperplane_Errors.txt', 'a+');
err.write('\n\n')
err.write(datafile);
# Labels Only
labels = [];
for label in train_labels:
    labels.append(train_labels.get(label));
# Progam
odata = [];
for i in range(0, num_rows, 1):
    if(train_labels.get(i) != None):
        odata.append(data[i]);

ntrain = [];
planes = [10, 100, 1000, 10000];
for k in planes:
    print('\nFor K = {} Random Planes:'.format(k));
    for i in range(0, k, 1):
        ltrain = [];
        w = [];
        for j in range(0, cols, 1):
            w.append(0);
        for j in range(0, cols, 1):
            w[j] = w[j] + rnd.uniform(1, -1);
        for i in range(0, num_rows, 1):
            if(train_labels.get(i) != None):
                dp = 0;
                dp = dotproduct(w, data[i]);
                s = int(math.copysign(1, dp));
                v = int((s+1)/2);
                ltrain.append(v);
        ntrain.append(ltrain);
        ntraint = zip(*ntrain);
        traindata = [];
        for r in ntraint:
            traindata.append(r);
    clf = svm.LinearSVC(C = 0.1, max_iter = 10000);
    scr = model_selection.cross_val_score(clf, traindata, labels, cv = 5);
    scr[:] = [1 - x for x in scr];

    oscr = model_selection.cross_val_score(clf, odata, labels, cv = 5);
    oscr[:] = [1 - x for x in oscr];

    print('Error for the New Features Data is= {}\nMean Error New Features {}'.format(scr, scr.mean()));
    print('Error for the Original Features Data is= {}\nMean Error Original Features Data {}'.format(oscr, oscr.mean()));


    err.write('\n\nFor K = {} Random Planes:'.format(k));
    err.write('\nError for the New Features Data is {}\nMean Error for New Features is {}'.format(scr, scr.mean()))
    err.write('\nError for the Original Features Data is {}\nMean Error for Original Features is {}'.format(oscr, oscr.mean()))
err.close();'''


test=[]
train=[]
train_new=[]
test_new=[]
trainlabels=[]
for i in range(0,num_rows,1):
	if train_labels.get(i)==None:
		test.append(data[i])
	else:
		train.append(data[i])
		if train_labels.get(i)==-1:
			trainlabels.append(0)
		else:	
			trainlabels.append(1)

for i in range(0,num_rows,1):
	new_data.append([])
	
clf = LinearSVC(max_iter=10000)
clf.fit(train, trainlabels)
predictions = clf.predict(test)
j=0
for i in range(0,num_rows,1):
	if train_labels.get(i)==None:
		#print(predictions[j],i)
		j=j+1
		
for i in range(0,k_in,1):
	w=[]
	for j in range(0,cols,1):
		w.append(rnd.uniform(1,-1))
	min=1000000000
	max=0
	for k in range(0,num_rows,1):
		dp=dot_product(w,data[k])
		if dp>max:
			max=dp
		if dp<min:
			min=dp
	w0=rnd.uniform(max,min)
	w.append(w0)
	for k in range(0,num_rows,1):
		dp=dot_product(w,datax[k])
		if dp<0:
			new_data[k].append(0)
		else:
			new_data[k].append(1)
for i in range(0,num_rows,1):
	if train_labels.get(i)==None:
		test_new.append(new_data[i])
	else:
		train_new.append(new_data[i])
clf = LinearSVC(max_iter=10000)
clf.fit(train_new, trainlabels)
predictions = clf.predict(test_new)
j=0
#print("predication for old = ",predictions)
for i in range(0,num_rows,1):
	if train_labels.get(i)==None:
		print(predictions[j],i)
		j=j+1
#print("predication for new = ",train_new)

