import Queue as queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import glob
x_pos=637
top_y=40
down_y=600
#f=open("a.txt","w+",0)
#f.write("y,label\n")
def countif(data):
    key=1 if data[0]==1 else 0
    count=sum(1 for item in data if item==key)
    staus=-1
    if key==1:
        staus=1 if count==16 else 2
    else:
        staus=3 if count==16 else 4
    return staus
def countif1(data):
    key=1 if data[0]==1 else 0
    count=sum(1 for item in data if item==key)
    count1=[i for i,v in enumerate(data) if v==key]
    staus=-1
    if key==1:
	if count==16:
		staus=1
        elif count1[-1]==8-1:
	#elif count==8:
		staus=2
        #staus=1 if count==16 else 2
    else:
	if count==16:
		staus=3
        elif count1[-1]==8-1:
	#elif count==8:
		staus=4
	else:
		staus=3
        #staus=3 if count==16 else 4
    return staus
def GeneralEquation(first_x,first_y,second_x,second_y,y):
    A=second_y-first_y
    B=first_x-second_x
    C=second_x*first_y-first_x*second_y
    x=-1*(y*B+C)/A
    print x
    x=1 if x>x_pos else 0
    return x
def direction(x,y):
    lr=[]
    tw=[]
    lrc=-1
    twc=-1
    com=-1
    tw_v=-1
    for i in x:
        lr.append(1 if i>x_pos else 0)
    dy1=y[1]-y[0]
    v1 = y[1]
    tw_v=1 if dy1>0 else 0
    tw.append(tw_v)
    tw.append(tw_v)
    for i in range(len(y[2:])):
        dy1 = y[i+2] -v1 
        if dy1>0:
            tw_v=1
        elif dy1<0:
            tw_v=0
        tw.append(tw_v)
        v1=y[i+2]
    lrc=countif(lr)
    twc=countif1(tw)
    if tw[0]!=-1:
       com=(lrc-1)*4+twc
    else: com=-1
    '''
    if com==2 or com==4 or com==6 or com==8 or com==10 or com==12 or com==14 or com==16:
	f.write("{}".format(y))     	
	f.write("{}\n".format(com))
        print com
    '''
    if com==2 or com==6:
    	if GeneralEquation(x[7],y[7],x[15],y[15],top_y):
		com=2
	else:
		com=6
    elif com==4 or com==8:
    	if GeneralEquation(x[7],y[7],x[15],y[15],down_y):
		com=4
	else:
		com=8
    elif com==10 or com==14:
    	if GeneralEquation(x[7],y[7],x[15],y[15],top_y):
		com=14
	else:
		com=10
    elif com==12 or com==16:
    	if GeneralEquation(x[7],y[7],x[15],y[15],down_y):
		com=16
	else:
		com=12
    return com
'''
for index in range(1,82):
        imglist=[]
        x=[]
        y=[]
        name=[]
        output_pics_path = glob.glob("GroundTruth/Clip" + str(index) +"/*.png")
        print(len(output_pics_path))
        for file in output_pics_path:         
            img=cv2.imread(file)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)
'''

