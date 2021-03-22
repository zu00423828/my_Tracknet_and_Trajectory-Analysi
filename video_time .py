import argparse
import Models
import Queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
import trackballclass as tc
#parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default = "")
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()
input_video_path =  args.input_video_path
output_video_path =  args.output_video_path
save_weights_path = args.save_weights_path
n_classes =  args.n_classes

if output_video_path == "":
	#output video in same path
	output_video_path = input_video_path.split('.')[0] + "_TrackNet.mp4"

#get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#start from first frame
currentFrame = 0

#width and height in TrackNet
width , height = 512, 288
img, img1, img2 = None, None, None

#load TrackNet model
modelFN = Models.TrackNet.TrackNet
m = modelFN( n_classes , input_height=height, input_width=width   )
m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
m.load_weights(  save_weights_path  )

# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames 
q = Queue.deque()
for i in range(0,8):
	q.appendleft(None)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height+320))


#both first and second frames cant be predict, so we directly write the frames to output video
#capture frame-by-frame
video.set(1,currentFrame); 
ret, img1 = video.read()
if not ret:
	print 'yes'
#write image to video
output_video.write(img1)
currentFrame +=1
#resize it 
img1 = cv2.resize(img1, ( width , height ))
#input must be float type
img1 = img1.astype(np.float32)

#capture frame-by-frame
video.set(1,currentFrame);
ret, img = video.read()
#write image to video
output_video.write(img)
currentFrame +=1
#resize it 
img = cv2.resize(img, ( width , height ))
#input must be float type
img = img.astype(np.float32)

x_list=[]
y_list=[]
classlabel={'2':0,'4':0,'6':0,'8':0,'10':0,'12':0,'14':0,'16':0}
label=['2','4','6','8','14','16','10','12']
#center=((1050,250),(1050,50),(750,250),(750,50),(450,250),(450,50),(140,250),(140,50))
center=((1050,250),(1050,80),(750,250),(750,80),(450,250),(450,80),(140,250),(140,80))
#outimg=cv2.imread('material/out.jpg')
tStart=0
while(True):
	#outimg=cv2.imread('material/out1.jpg')
	img2 = img1
	img1 = img

	#capture frame-by-frame
	video.set(1,currentFrame); 
	ret, img = video.read()

	#if there dont have any frame in video, break
	if not ret: 
		break

	#img is the frame that TrackNet will predict the position
	#since we need to change the size and type of img, copy it to output_img
	output_img = img

	#resize it 
	img = cv2.resize(img, ( width , height ))
	#input must be float type
	img = img.astype(np.float32)


	#combine three imgs to  (width , height, rgb*3)
	X =  np.concatenate((img, img1, img2),axis=2)

	#since the odering of TrackNet  is 'channels_first', so we need to change the axis
	X = np.rollaxis(X, 2, 0)
	#prdict heatmap
	pr = m.predict( np.array([X]) )[0]

	#since TrackNet output is ( net_output_height*model_output_width , n_classes )
	#so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
	#.argmax( axis=2 ) => select the largest probability as class
	pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2 )

	#cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
	pr = pr.astype(np.uint8) 

	#reshape the image size as original input image
	heatmap = cv2.resize(pr  , (output_width, output_height ))

	#heatmap is converted into a binary image by threshold method.
	ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)

	#find the circle in image with 2<=radius<=7
	circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)

	#In order to draw the circle in output_img, we need to used PIL library
	#Convert opencv image format to PIL image format
	PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)   
	PIL_image = Image.fromarray(PIL_image)

	#check if there have any tennis be detected
	com=-1
	com1=-1
	if circles is not None:
		#if only one tennis be detected
		if len(circles) == 1:

			x = int(circles[0][0][0])
			y = int(circles[0][0][1])
			print currentFrame, x,y
			x_list.append(x)
			y_list.append(y)
			if len(x_list)==16:
				com=tc.direction(x_list,y_list)
				tEnd = time.time()
				print "It cost %f sec" % (tEnd - tStart)
				print tEnd - tStart
				tStart = time.time()
				if com1!=com:
					com1=com
					for key in classlabel:
						if key==str(com1):
							classlabel[key]+=1
							#print key+ ":" +str(classlabel[key])+"\n"
				del x_list[0]
				del y_list[0]
			#push x,y to queue
			q.appendleft([x,y])   
			#pop x,y from queue
			q.pop()    
		else:
			#push None to queue
			q.appendleft(None)
			#pop x,y from queue
			q.pop()
	else:
		#push None to queue
		q.appendleft(None)
		#pop x,y from queue
		q.pop()
	#draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
	for i in range(0,8):
		if q[i] is not None:
			draw_x = q[i][0]
			draw_y = q[i][1]	
			bbox =  (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
			draw = ImageDraw.Draw(PIL_image)
			draw.ellipse(bbox, outline ='yellow')
			del draw

	#Convert PIL image format back to opencv image format
	opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

	index=0
	for key in label:
		x1=center[index][0]+100+10
    		y1=center[index][1]+20+10
		cv2.putText(outimg,str(classlabel[key]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
		
		#opencvImage=cv2.putText(opencvImage,"Label"+key+" : "+str(classlabel[key]),(50,50+count),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
		index=index+1
	#write image to output_video
	tmp2=np.vstack((outimg,opencvImage)) 
	output_video.write(tmp2)
	#output_video.write(opencvImage)

	#next frame
	currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()
print "finish"

'''
for key in classlabel:
	if key==str(2):
		print(classlabel[key])
		classlabel[key]+=1
for key in classlabel:
	print "key:%s,value:%d" %(key,classlabel[key])
'''
