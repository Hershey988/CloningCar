#!/usr/bin/env python

import pandas as pd # Needed to read CSV file
import os # Needed to find the path to the data
import ntpath # Needed to split the path
import matplotlib.pyplot as plt # Needed to plot data
import numpy as np # Needed for histogram and other stuff
import random # Needed to shuffle the data array
import csv # Needed to write to a csv file
from skimage import io
import cv2
from PIL import Image

def splitPath(data):
    front, end = ntpath.split(data)
    # return the name of the image file
    return "IMG/" + end
    
# Helper function that generates coordinates for polygons.
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


# Helper function to add shadow to images
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


# GENERATE SHADOW IMAGES AND ADD THEM TO THE EXCEL SHEET and SAVE THEM
def add_shadow_csv(data_dir):
    
    #data_dir = 'CloningCar/udacityData'
    #image_dir = 'CloningCar/udacityData/'
    data_to_process = pd.read_csv(os.path.join(data_dir, 'balanced_data.csv'))
    print('IN add_shadow_csv:  ', data_to_process.shape )
    with open(os.path.join(data_dir, 'balanced_data.csv'),'a') as output:
        mywrite = csv.writer(output)
        for i in range(len(data_to_process['steering'])):
            #adding shadows to the left image
      
            #Check where the image belongs to
            path_to_image = ''
            output_path = ''

            if os.path.isfile(os.path.join('udacityData',data_to_process['image'][i])):
                path_to_image = os.path.join('udacityData', data_to_process['image'][i])
                output_path = os.path.join('udacityData', data_to_process['image'][i].replace('.jpg', '_shadow.jpg').strip())
            else:
                path_to_image = os.path.join('finalData', data_to_process['image'][i])
                output_path = os.path.join('finalData', data_to_process['image'][i].replace('.jpg', '_shadow.jpg').strip())
      
            #ADD SHADOW TO THE IMAGE and save it
            img = io.imread(path_to_image)
            img_shadow = add_shadow(img,i % 3)#adding upto 3 shadows
            Image.fromarray(img_shadow).save(output_path)
        
            #WRITE IT TO THE CSV FILE.
            mywrite.writerow([data_to_process['image'][i].replace('.jpg', '_shadow.jpg').strip(), data_to_process['steering'][i]])




def plotRawData(fd, fd_2):
    # Plotting the steering angle data
    i = np.linspace(0, 15539,15539)
    #fd['steering'] = pd.to_numeric(fd['steering'])
    #steering_list = np.asarray(fd['steering'])
    steering_list = pd.concat([fd['steering'], fd_2['steering']])
    
    print(steering_list.shape)
    
    plt.title('Steering Angles vs. Data Points')
    plt.xlabel('Steering Angles')
    plt.ylabel('Data Points')

    plt.plot(steering_list, i,'o')
    plt.show()
    '''Another way is to combine a set of steering angles in certain range
        and group them together'''

    num_of_buckets = 50       # Number of different groups(bucket) we want to create.
    samples_per_bucket = 200    # Number of sample data per group(bucket).


    # Generate a histogram and group the steering angles
    histogram, buckets = np.histogram(steering_list, num_of_buckets)
    # Print the amount of data per bucket
    plt.title('Number of Steering Angle Data Points Specific to Each Bucket Range')
    plt.xlabel('Bucketed Steering Angles')
    plt.ylabel('Number of Steering Angle Data Points')
    print(histogram)
    # Find the center of the distribution.
    center = (buckets[:-1] + buckets[1:])/2.0
    # Plot the histogram as a bar graph
    plt.bar(center,histogram, width=0.01) # Change the width to see better
    plt.show()


def balanceData(fd, fd_2, data_dir):
    data_to_process = pd.concat([fd.head(fd.shape[0]), fd_2.head(fd_2.shape[0])], ignore_index=True)
    #print(data_to_process.tail(35))
    #print(data_to_process.shape)
    
    num_of_buckets = 50         # Number of different groups(bucket) we want to create.
    samples_per_bucket = 300    # Number of sample data per group(bucket).
    histogram, buckets = np.histogram(data_to_process['steering'], num_of_buckets)
    
    # list that will hold all the indices to be deleted from the original dataset
    indices_to_delete = []

    # Go through the entire data set and delete data for each bucket that has more than N samples
    for b_i in range(len(buckets) - 1):
        # list that will contain the index of sample data for each bucket.
        bucket_list_indices = []
        # There is probably a better way to do this on the entire list...
        steering_val = 0.0
        for i in range(len(data_to_process['steering'])):
            steering_val = data_to_process['steering'][i]
            # Check if the steering angle belongs to this bucket
            if (steering_val >= buckets[b_i]) and (steering_val <= buckets[b_i + 1]):
                bucket_list_indices.append(i)

        '''Now we have the list of samples for bucket b_i. Need to remove samples if
        the len(bucket_list_indices) is greater than the number of samples per bucket
        allowed. Before we add the indices to the delete list, need to make sure we
        delete the datapoints randomly. The bucket_list_indices will contain indices
        in increasing order therefore we need to shuffle the list so that we truly
        delete data at random. '''
        print(len(bucket_list_indices))
        bucket_list_indices = random.sample(bucket_list_indices, len(bucket_list_indices))
        # get all the indices after the bucket_list_indices[samples_per_bucket]
        bucket_list_indices = bucket_list_indices[samples_per_bucket:]
        # append the list for the bucket the overall data to be removed list
        indices_to_delete.extend(bucket_list_indices)
  
    ''' https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-index-labels/ '''
    # Finally drop the data we do not need....
    data_to_process.drop(data_to_process.index[indices_to_delete], inplace=True)
    print(data_to_process.shape)
    
    # Need to reset the indexing otherwise we can not access the data properly:
    # You could dump the file and reopen it to get the index, but idk how that will work
    data_to_process.reset_index(inplace = True)

    print(data_to_process.shape)


    with open(os.path.join(data_dir, 'balanced_data.csv'),'w') as output:
        mywrite = csv.writer(output)
        mywrite.writerow(['image', 'steering'])
        for i in range(len(data_to_process['steering'])):
            #print(data_to_process['center'][i], data_to_process['steering'][i])
            mywrite.writerow([data_to_process['center'][i].strip(), data_to_process['steering'][i]])
            mywrite.writerow([data_to_process['left'][i].strip(), data_to_process['steering'][i]])
            mywrite.writerow([data_to_process['right'][i].strip(), data_to_process['steering'][i]])
    
    # Now that a balanced sheet was created, we will generate shadows for each of these
    # images.
    add_shadow_csv(data_dir)
    ##END ADDED

    some_fd = pd.read_csv(os.path.join(data_dir, 'balanced_data.csv'))
    print('AFTER SHADOWS:: add_shadow_csv:  ', some_fd.shape)

    #data_to_process.to_csv(os.path.join(data_dir, 'balanced_data.csv'))

    # histogram and group the steering angles
    histogram, buckets = np.histogram(data_to_process['steering'], num_of_buckets)
    # Print the amount of data per bucket
    plt.title('Balanced Data: Number of Steering Angle Data Points Specific to Each Bucket Range')
    plt.xlabel('Bucketed Steering Angles')
    plt.ylabel('Number of Steering Angle Data Points')
    print(buckets)
    # Find the center of the distribution.
    center = (buckets[:-1] + buckets[1:])/2.0
    print(center)
    # Plot the histogram as a bar graph
    plt.bar(center,histogram, width=0.01) # Change the width to see better
    plt.show()


#Main function of the file.
def main():
    udacity_data_dir = 'udacityData'
    our_data_dir = 'finalData'
    data_dir = 'newSpreadsheet'
    # read the csv file
    fd = pd.read_csv(os.path.join(udacity_data_dir, 'driving_log.csv'))
    
    col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    fd_2 = pd.read_csv(os.path.join(our_data_dir, 'driving_log_new.csv'), names = col_names)
    
    # plot the raw data from both directories.
    #plotRawData(fd, fd_2)
    
    #Do Some Stripping
    fd_2['center'] = fd_2['center'].apply(splitPath)
    fd_2['left'] = fd_2['left'].apply(splitPath)
    fd_2['right'] = fd_2['right'].apply(splitPath)
    
    #print(fd_2.head(30))
    balanceData(fd, fd_2, data_dir)

if __name__ == "__main__":
    main()

