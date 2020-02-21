#!/usr/bin/env python

import pandas as pd # Needed to read CSV file
import os # Needed to find the path to the data
import ntpath # Needed to split the path
import matplotlib.pyplot as plt # Needed to plot data
import numpy as np # Needed for histogram and other stuff
import random # Needed to shuffle the data array
import csv # Needed to write to a csv file

data_dir = 'udacityData'
# read the csv file
fd = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))


# Plotting the steering angle data
i = np.linspace(0, 8035,8036)
#fd['steering'] = pd.to_numeric(fd['steering'])

plt.title('Steering Angles vs. Data Points')
plt.xlabel('Steering Angles')
plt.ylabel('Data Points')

plt.plot(fd['steering'], i,'o')
plt.show()
'''Another way is to combine a set of steering angles in certain range
   and group them together'''

num_of_buckets = 50       # Number of different groups(bucket) we want to create.
samples_per_bucket = 200    # Number of sample data per group(bucket).


# Generate a histogram and group the steering angles
histogram, buckets = np.histogram(fd['steering'], num_of_buckets)
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



# We need to create an array/list that stores the randomly selected data
# Copy the pointer of CSV file
data_to_process = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'))

# list that will hold all the indices to be deleted from the original dataset
indices_to_delete = []

# Go through the entire data set and delete data for each bucket that has more than 200 samples
for b_i in range(len(buckets) - 1):
  # list that will contain the index of sample data for each bucket.
  bucket_list_indices = []
  # There is probably a better way to do this on the entire list...
  steering_val = 0
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
  bucket_list_indices = random.sample(bucket_list_indices, len(bucket_list_indices))
  # get all the indices after the bucket_list_indices[samples_per_bucket]
  bucket_list_indices = bucket_list_indices[samples_per_bucket:]
  # append the list for the bucket the overall data to be removed list
  indices_to_delete.extend(bucket_list_indices)
  
''' https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-index-labels/ '''
# Finally drop the data we do not need....
data_to_process.drop(data_to_process.index[indices_to_delete], inplace=True)

#Need to reset the indexing otherwise we can not access the data properly:
# You could dump the file and reopen it to get the index, but idk how that will work
data_to_process.reset_index(inplace = True)




with open(os.path.join(data_dir, 'balanced_data.csv'),'w') as output:
    mywrite = csv.writer(output)
    mywrite.writerow(['image', 'steering'])
    for i in range(len(data_to_process['steering'])):
        #print(data_to_process['center'][i], data_to_process['steering'][i])
        mywrite.writerow([data_to_process['center'][i].strip(), data_to_process['steering'][i]])
        mywrite.writerow([data_to_process['left'][i].strip(), data_to_process['steering'][i]])
        mywrite.writerow([data_to_process['right'][i].strip(), data_to_process['steering'][i]])

##END ADDED


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


