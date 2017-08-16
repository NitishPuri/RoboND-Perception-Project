#!/usr/bin/env python
import numpy as np
import pickle
import rospy

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    models = [\
       'biscuits',
       'book',
       'create',
       'eraser',
       'glue',
       'snacks',
       'soap',
       'soap2',
       'soda_can',
       'sticky_notes'
       ]

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []
    
    number_of_poses = 500

    print("Starting to roll,...")

    for model_name in models:
        spawn_model(model_name)
        print("Now rolling,... : {}".format(model_name))
        for i in range(number_of_poses):
            print("Attempt number,.. : {} - {}".format(model_name , i+1))
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            sample_cloud = None
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            print("Computing histograms.,,.")
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)                        
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        print("Delete the added model of,... {} ".format(model_name))
        delete_model()
        print("Succesfully deleted model of,... {} ".format(model_name))

    print("Dumping data to 'training_set.sav'")
    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

