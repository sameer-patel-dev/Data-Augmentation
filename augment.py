import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np

Folder_name="Dataset_AP"
Extension=".jpg"

def resize_image(image,w,h):
    image=cv2.resize(image,(w,h))
    cv2.imwrite(Folder_name+"/Resize-24"+str(w)+str(h)+Extension, image)

def crop_image(image,y1,y2,x1,x2):
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Folder_name+"/Crop-24"+str(x1)+str(x2)+str(y1)+str(y2)+Extension, image)

def padding_image(image,topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0]):
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Folder_name + "/padd-24" + str(topBorder) + str(bottomBorder) + str(leftBorder) + str(rightBorder) + Extension, image)


def invert_image(image,channel):
    # image=cv2.bitwise_not(image)
    image=(channel-image)
    cv2.imwrite(Folder_name + "/invert-24"+str(channel)+Extension, image)

def add_light(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light-24"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark-24" + str(gamma) + Extension, image)

def add_light_color(image, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Folder_name + "/light_color-24"+str(gamma)+Extension, image)
    else:
        cv2.imwrite(Folder_name + "/dark_color-24" + str(gamma) + Extension, image)

def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/saturation-24" + str(saturation) + Extension, image)


def multiply_image(image,R,G,B):
    image=image*[R,G,B]
    cv2.imwrite(Folder_name + "/Multiply-24" + str(R) + str(G) + str(B) + Extension, image)

def gausian_blur(image,blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Folder_name+"/GausianBLur-24"+str(blur)+Extension, image)


def bileteralBlur(image,d,color,space):
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(Folder_name + "/BileteralBlur-24"+str(d)+str(color)+str(space)+ Extension, image)

def erosion_image(image,shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Erosion-24"+str(shift) + Extension, image)

def dilation_image(image,shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Folder_name + "/Dilation-24"+ str(shift)+ Extension, image)

def opening_image(image,shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Folder_name + "/Opening-24"+ str(shift)+ Extension, image)

def closing_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Folder_name + "/Closing-24"+ str(shift) + Extension, image)

def morphological_gradient_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Folder_name + "/Morphological_Gradient-24" +str(shift) + Extension, image)

def top_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name + "/Top_Hat-24" +str(shift) + Extension, image)

def black_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Folder_name + "/Black_Hat-24" + str(shift) + Extension, image)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name+"/Sharpen-24"+Extension, image)


def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Folder_name + "/Addeptive_gaussian_noise-24" + Extension, image)

def salt_image(image,p,a):
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Folder_name + "/Salt-24"+str(p)+str(a) + Extension, image)

def paper_image(image,p,a):
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Paper-24" + str(p) + str(a) + Extension, image)

def salt_and_paper_image(image,p,a):
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1

    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Folder_name + "/Salt_And_Paper-24" + str(p) + str(a) + Extension, image)


def grayscale_image(image):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/Grayscale-24" + Extension, image)




image_file="Dataset/24.jpg"
image=cv2.imread(image_file)

resize_image(image,450,400)

crop_image(image,100,400,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,400,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,0,350)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,0,300,100,450)#(y1,y2,x1,x2)(bottom,top,left,right)
crop_image(image,100,300,100,350)#(y1,y2,x1,x2)(bottom,top,left,right)

padding_image(image,100,0,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,100,0,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,100,0)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,0,0,0,100)#(y1,y2,x1,x2)(bottom,top,left,right)
padding_image(image,100,100,100,100)#(y1,y2,x1,x2)(bottom,top,left,right)


invert_image(image,255)
invert_image(image,250)
invert_image(image,300)

add_light(image,1.5)
add_light(image,2.0)
add_light(image,2.5)
add_light(image,3.0)
add_light(image,4.0)
add_light(image,5.0)
add_light(image,0.7)
add_light(image,0.4)
add_light(image,0.3)

add_light_color(image,255,1.5)
add_light_color(image,50,2.0)
add_light_color(image,255,0.7)

saturation_image(image,50)
saturation_image(image,75)
saturation_image(image,100)


multiply_image(image,0.5,0.5,0.5)
multiply_image(image,0.25,0.25,0.25)
multiply_image(image,1.25,1.25,1.25)
multiply_image(image,1.5,1.5,1.5)


gausian_blur(image,0.25)



bileteralBlur(image,9,75,75)
bileteralBlur(image,12,100,100)
bileteralBlur(image,25,100,100)
bileteralBlur(image,40,75,75)

erosion_image(image,1)
erosion_image(image,2)

dilation_image(image,1)


opening_image(image,1)
opening_image(image,2)

closing_image(image,1)
closing_image(image,2)

morphological_gradient_image(image,2)

top_hat_image(image,200)
top_hat_image(image,300)
top_hat_image(image,500)

black_hat_image(image,200)
black_hat_image(image,300)
black_hat_image(image,500)

sharpen_image(image)

addeptive_gaussian_noise(image)

salt_image(image,0.5,0.009)
paper_image(image,0.5,0.009)


salt_and_paper_image(image,0.5,0.009)

grayscale_image(image)
