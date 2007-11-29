import numpy

# used ocropus/tesseract on original data file
ssa = '2633 2747 2868 3042 3278 3574 3689 4067 4419 4910 5481 5684 5895 6498 6891 8065 8745 9156 10287 11099 12629 14306 15835 16916 18200'
hcfa = '8474 9175 10142 11104 12648 14340 15918 17162 19278 21568 25181 27931'
initial_x = '2582 2726 2874 3055 3275 3521 3753 4075 4443 4873 5312 5647 6001 6504 7073 7871 8566 9261 10212 11250 12661 14228 15752 17194 19073 21733 24741 27573'
initial_P = '67 66 65 65 65 65 65 65 65 65 65 65 65 65 65 64 54 53 53 53 53 53 53 53 54 64 68 80'
MLE_x = '2541 2711 2864 3045 3269 3519 3736 4063 4433 4876 5331 5644 5972 6477 7032 7866 8521 9198 10160 11159 12645 14289 15835 17171 19106 21675 25027 27932 '
MLE_P='178 185 186 186 186 186 186 186 186 186 186 186 186 186 185 179 110 108 108 108 108 108 108 108 109 119 120 129 '


year = numpy.arange(1949,1977)
ssa = numpy.array(map(int,ssa.strip().split()))
hcfa = numpy.array(map(int,hcfa.strip().split()))
initial_x = numpy.array(map(int,initial_x.strip().split()))
initial_P = numpy.array(map(int,initial_P.strip().split()))
MLE_x = numpy.array(map(int,MLE_x.strip().split()))
MLE_P = numpy.array(map(int,MLE_P.strip().split()))

nan = numpy.nan
ssa = numpy.array(list(ssa) + [ nan, nan, nan])
hcfa = numpy.array([nan]*16 + list(hcfa))

fd = open('table1.csv',mode='w')
print >>fd, 'year,ssa,hcfa,initial_x,initial_P,MLE_x,MLE_P'
for i in range(len(year)):
    print >>fd,'%d,%s,%s,%d,%d,%d,%d'%(year[i], str(ssa[i]), str(hcfa[i]), initial_x[i], initial_P[i], MLE_x[i], MLE_P[i])
fd.close()

r=[1,2,3,4,5,10,20,40,50,75]
u=[2500,2417,2396,2383,2374,2342,2279,2277,2276,2277]
p=[1.1,1.114,1.116,1.116,1.116,1.116,1.116,1.116,1.116,1.116]
Q=[10000,49837, 78153,93513,100571, 105152, 104814, 105115, 105097,105115]
R11=[10000,41583,54666,59558,62483,65725,67760,68636,68663,68675]
R22=[10000,24105,25486,25580,25384,23920,20971,19394,19354,19329]
ll=[885,680,675,675,674,674,674,672,671,671,671]

fd = open('table2.csv',mode='w')
print >>fd, 'r,u,phi,Q,R11,R22,ll'
for i in range(len(r)):
    r[i]
    u[i]
    p[i]
    Q[i]
    R11[i]
    R22[i]
    ll[i]
    print >>fd,'%d,%d,%.3f,%d,%d,%d,%d'%(r[i], u[i], p[i], Q[i], R11[i], R22[i], ll[i])
fd.close()

