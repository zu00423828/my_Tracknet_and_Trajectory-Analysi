import numpy as np
import cv2

#上方圖片合成
img1 = np.zeros((320,1280,3), np.uint8)
img1.fill(50)#黑底
#img1.fill(255)#白底

#cv2.seamlessClone(src, dst, mask, center, flags) 合併在圖檔中
#src 目標 , dst 背景, mask 目標位置, center 產生位置, flags
#选择融合的方式，目前有NORMAL_CLONE、MIXED_CLONE和MONOCHROME_TRANSFER三种方法
center=((1050,250),(1050,50),(750,250),(750,50),(450,250),(450,50),(140,250),(140,50))
for i in range(8):
    src=cv2.imread('material2/'+str(i+1)+'.jpg')
    cv2.resize(src,None, fx=1.5, fy=1.5)
    if i==0:
        out=img1   
    #src_mask = np.zeros(src.shape, src.dtype)
    src_mask=255 * np.ones(src.shape, src.dtype)
    out=cv2.seamlessClone(src,out,src_mask,center[i],cv2.NORMAL_CLONE)
    x=center[i][0]+60
    y=center[i][1]+20
    cv2.putText(out,"=",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
cv2.imwrite('out.jpg',out)

#影片 上下合成
outimg=cv2.imread('out.jpg')
count=1
for i in range(8):
    x=center[i][0]+100
    y=center[i][1]+20
    cv2.putText(outimg,str(count),(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
cap=cv2.VideoCapture("Clip1.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output2_9.mp4',fourcc,30,(1280, 720+320))
i=0
while(cap.isOpened()):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()
  if ret == True:
      # 顯示圖片
      tmp2=np.vstack((outimg,frame)) #縱向合併
      #cv2.imshow('frame', frame)
      cv2.imwrite('frame/result'+str(i)+'.jpg',tmp2)
      out.write(tmp2)
      print(i)
      i=i+1
  else:
      break
 
    
# 釋放攝影機
cap.release()
out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()

