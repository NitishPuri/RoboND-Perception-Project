#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Helper function to perform Statistical Outlier removal on the input cloud 
def remove_outliers(in_cloud):
    fil = in_cloud.make_statistical_outlier_filter()
    fil.set_mean_k(5)
    fil.set_std_dev_mul_thresh(0.5)
    out_cloud = fil.filter()
    print("No. of points after outlier removal filtering,.. {}".format(out_cloud.size))
#    pcl.save(out_cloud, "fil_inliers.pcd")
    fil.set_negative(True)
    pcl.save(fil.filter(), "fil_outliers.pcd")
    
    return out_cloud
    
# Helper function to perform Voxel Grid Downsampling on the input cloud
def voxDownsampling(in_cloud):
    vox = in_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    out_cloud = vox.filter()
#    pcl.save(out_cloud, "downsampledPoints.pcd")
    print("Vox grid Downsampled point cloud of length ,... {}".format(out_cloud.size))
    return out_cloud

# Helper function to perform Passthrough filter on the input, 
# This filter is modeled to correctly passthrough only the relevant table and objects 
def passthroughFilter(in_cloud):
    passthrough = in_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.60
    axis_max = 1.0
    passthrough.set_filter_limits(axis_min, axis_max)
    out_cloud = passthrough.filter()
    print("Passthrough filter applied on z({},{}),.. no. of points ,.. : {}".format(axis_min, axis_max, out_cloud.size))
    # Passthrough filter on 'y'
    passthrough = out_cloud.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    out_cloud = passthrough.filter()
    print("Passthrough filter applied on y({},{}),.. no. of points ,.. : {}".format(axis_min, axis_max, out_cloud.size))
#    pcl.save(out_cloud, "passthrough.pcd")    
    return out_cloud

# Perform RANSAC filtering on the cloud to filter out the table and return the remaining objects
def RansacFilter(in_cloud):
    seg = in_cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    print("RANSAC Place Segmentation,.....")

    # TODO: Extract inliers and outliers
    cloud_table = in_cloud.extract(inliers, negative=False)
    cloud_objects = in_cloud.extract(inliers, negative=True)

    print("Number of points in Table Cloud : {}".format(cloud_table.size))
    print("Number of points in Objects Cloud : {}".format(cloud_objects.size))

    # Publish ROS messages    
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)

#    pcl.save(cloud_table, "table.pcd")
#    pcl.save(cloud_objects, "objects.pcd")
    
    return cloud_objects



# Cluster the points and return an colored cloud to visualize eacjh cluster properly
def EuclideanClustering(in_cloud):
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(in_cloud)    
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(2000)

    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    #Assign a color corresponding to each segmented object in scene ??????
#    get_color_list.color_list = []
    cluster_color = get_color_list(len(cluster_indices))

    print ("Number of clusters : {}".format(len(cluster_indices)))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])


    #Create new cloud containing all clusters, each with unique color ???????
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    ros_cluster_cloud = pcl_to_ros(cluster_cloud)   

    # TODO: Publish ROS messages
    pcl_cluster_pub.publish(ros_cluster_cloud)

#    pcl.save(cluster_cloud, "clusters.pcd")
    return cluster_indices


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    print("\nReceived pcl_callback,...")
# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    print("No. of points in received cloud : {}".format(pcl_cloud.size))
    
    # Save the received cloud.,.
#    pcl.save(pcl_cloud, "receivedPoints.pcd")    
#    rospy.signal_shutdown("Task completed,..")    
#    return
    
    # TODO: Voxel Grid Downsampling
    cloud_filtered = voxDownsampling(pcl_cloud)
    
    # TODO: PassThrough Filter
    cloud_filtered = passthroughFilter(cloud_filtered)
        
    # TODO: RANSAC Plane Segmentation
    cloud_filtered = RansacFilter(cloud_filtered)

    # Removing outliers,..
    cloud_filtered_objects = remove_outliers(cloud_filtered)

    ros_temp_points = pcl_to_ros(cloud_filtered_objects)
    passthrough_pub.publish(ros_temp_points)

#    print
    # TODO: Euclidean Clustering
    cluster_indices = EuclideanClustering(cloud_filtered_objects)

# Exercise-3 TODOs:

    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_filtered_objects.extract(pts_list)

        # TODO: Convert the cluster from pcl to ROS using helper functions
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)                        

        # Compute the associated feature vector
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(cloud_filtered_objects[pts_list[0]])[0:3]
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    pick_labels = []
    pick_centroids = []
    
    test_scene_num = Int32()
    test_scene_num.data = 1
    
    pick_list_num = 1
    
    dict_list = []
    
    yaml_filename = "output{}_{}.yaml".format(test_scene_num.data, pick_list_num)
    
    
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map


    print("Iterating over objects in the pick list,...")
    # TODO: Loop through the pick list
    for pick_object_param in object_list_param:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pick_object_name = pick_object_param['name']
        pick_object_group = pick_object_param['group']
        
        print("Looking for [[{}]] to be placed in [[{}]] box.".format(pick_object_name, pick_object_group))
        
        object_name = String()
        object_name.data = pick_object_name
        
        # Index of the object to be picked in the `detected_objects` list
        pick_object_index = None
        for i, detected_object in enumerate(object_list):
            print("{} :: {} ".format(i, detected_object.label))
            if(detected_object.label == pick_object_name):
                pick_object_index = i
                break
            
        if(pick_object_index == None):
            print("ERROR:::: {} not found in the detected object list".format(pick_object_name))
            continue
            
#        print("Found object at index {}".format(pick_object_index))
        
        points_arr = ros_to_pcl(object_list[pick_object_index].cloud).to_array()
#        print("Converted to array...")
        pick_object_centroid = np.mean(points_arr, axis=0)[:3] 
        print("Centroid found : {}".format(pick_object_centroid))

        pick_labels.append(pick_object_name)
        pick_centroids.append(pick_object_centroid)

        # Create pick_pose for the object
        pick_pose = Pose()
        pick_pose.position.x = pick_object_centroid[0]
        pick_pose.position.y = pick_object_centroid[1]
        pick_pose.position.z = pick_object_centroid[2]
#        pick_pose.orientation = 
                                                
        # TODO: Create 'place_pose' for the object
        place_pose = Pose()
        if(pick_object_group == 'green'):
            place_pose.position.x =  0
            place_pose.position.y = -0.71
            place_pose.position.z =  0.605
        else:
            place_pose.position.x =  0
            place_pose.position.y =  0.71
            place_pose.position.z =  0.605
                
#        print("Place pose created,... {}")

        # TODO: Assign the arm to be used for pick_place
        arm_name = String()
        if(pick_object_group == 'green'):
            arm_name.data = 'right'
        else:
            arm_name.data = 'left'
                    
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        
        dict_list.append(yaml_dict)
        
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            print("Creating service proxy,...")
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            print("Requesting for service reponse,...")
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(yaml_filename, dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    print("ROS node initialization")
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    print("Create Subscribers")
#    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    print("Create Publishers")
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size = 1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size = 1)
    passthrough_pub = rospy.Publisher("/received_points", PointCloud2, queue_size = 1)

    # TODO: Load Model From disk
    modelFileName = 'model.sav'
    print("Load Model From disk, {}".format(modelFileName))
    trained_model = pickle.load(open(modelFileName, 'rb'))
    clf = trained_model["classifier"]
    scaler = trained_model["scaler"]
    encoder = LabelEncoder()
    encoder.classes_ = trained_model["classes"]

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    print("Start spinning,.......")
    while not rospy.is_shutdown():
        rospy.spin()        
