
#This script will take a csv file  and adds new images with a shadow to the img file 
'''
1) make sure you change  datadir
2)make sure you change  imagedir
3) make sure you change  inputcsvfile

'''

import csv
import pandas as pd # Needed to read CSV file
import os # Needed to find the path to the data
import ntpath # Needed to split the path
from skimage import io
import cv2
from PIL import Image

def generate_shadow_coordinates(imshape, no_of_shadows):
  vertices_list=[]
  for index in range(no_of_shadows):
    vertex=[]
    for dimensions in range(np.random.randint(3,15)): 
      ## Dimensionality of the shadow polygon
      vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
      vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
      vertices_list.append(vertices)
  return vertices_list ## List of shadow vertices


def add_shadow(image,no_of_shadows):
  image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
  mask = np.zeros_like(image)
  imshape = image.shape
  image_RGB = image
  vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) 
  #3 getting list of shadow vertices
  for vertices in vertices_list:
    cv2.fillPoly(mask, vertices, 255) 
    ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5
    ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
  
  #print(image_RGB.shape)
  
  return image_RGB

data_dir = 'udacityData'
image_dir = 'udacityData'
data_to_process = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))
for i in range(len(data_to_process['steering'])):
  #adding shadows to the left image
  path = os.path.join(image_dir.strip(),data_to_process['left'][i].strip())
  img = io.imread(path)
  #print(img.shape)
  img_shadow = add_shadow(img,i % 3)#adding upto 9 shadows
  output_path = os.path.join(image_dir.strip(), data_to_process['left'][i].replace('.jpg', '_shadow.jpg').strip())
  #io.imsave(output_path, img_shadow)
  Image.fromarray(img_shadow).save(output_path)

  
  #adding shadows to the left image
  path = os.path.join(image_dir.strip(),data_to_process['center'][i].strip())
  img = io.imread(path)
  img_shadow = add_shadow(img,i % 3)#adding upto 9 shadows
  output_path = os.path.join(image_dir.strip(), data_to_process['center'][i].replace('.jpg', '_shadow.jpg').strip())
  #io.imsave(output_path, img_shadow)
  Image.fromarray(img_shadow).save(output_path)
  
  #adding shadows to the left image
  path = os.path.join(image_dir.strip(),data_to_process['right'][i].strip())
  img = io.imread(path)
  img_shadow = add_shadow(img,i % 3)#adding upto 9 shadows
  output_path = os.path.join(image_dir.strip(), data_to_process['right'][i].replace('.jpg', '_shadow.jpg').strip())
  Image.fromarray(img_shadow).save(output_path)





 





