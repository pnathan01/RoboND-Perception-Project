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

def pcl_to_ros(pcl_array):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = "world"

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg

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

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    detected_objects_labels = []
    detected_objects = []
    labeled_features = []

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
#    filename = "raw.pcd"
#    pcl.save(pcl_data, filename)
#    print("Raw Data")
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    x = 1.0
    outlier_filter.set_std_dev_mul_thresh(x)
    out_filtered = outlier_filter.filter()
#    filename = "filtered.pcd"
#    pcl.save(out_filtered, filename)
#    print("SOF Data")

    # TODO: Voxel Grid Downsampling
    vox = out_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.002
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    grid_filtered = vox.filter()
#    filename = "grid_filtered.pcd"
#    pcl.save(grid_filtered, filename)
#    print("Grid Filtered Data")

    # TODO: PassThrough Filter for table height
    passthrough = grid_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
#    axis_min = 0.6
#    axis_max = 1.3
    axis_min = 0.6
    axis_max = 1.2
    passthrough.set_filter_limits(axis_min, axis_max)
    passthro_filtered = passthrough.filter()

# Passthrough filter for y
    passthrough = passthro_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)
    passthro_filtered = passthrough.filter()

#    filter_axis = 'y'
#    passthrough.set_filter_field_name(filter_axis)
#    axis_min = 3.0
#    axis_max = 3.2
#    passthrough.set_filter_limits(axis_min, axis_max)
#    passthro_filtered = passthrough.filter()

#    filename = "passthro_filtered.pcd"
#    pcl.save(passthro_filtered, filename)
#    print("Passthro Data")

    # TODO: RANSAC Plane Segmentation
    seg = passthro_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_inliers = passthro_filtered.extract(inliers, negative=True)
    extracted_outliers = passthro_filtered.extract(inliers, negative=False)
#    filename = "inliers.pcd"
#    pcl.save(extracted_inliers, filename)
#    filename = "outliers.pcd"
#    pcl.save(extracted_outliers, filename)
#    print("In and Outlier Data")

    # Double Statistical Outlier Filtering
    outlier_filter = extracted_inliers.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(100)
    x = 2.0
    outlier_filter.set_std_dev_mul_thresh(x)
    out_filtered = outlier_filter.filter()
#    filename = "inlier_filtered.pcd"
#    pcl.save(out_filtered, filename)
#    print("Double SOF Data")

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(out_filtered)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(1500)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

#    print("Cluster Indices Length")
#    print(len(cluster_indices))

# Instantiate the class and then call the method to create cluster cloud from points list
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

#    filename = "Cluster_Cloud.pcd"
#    pcl.save(cluster_cloud, filename)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately

    # TODO: Convert PCL data to ROS messages
#    ros_objects_msg = pcl_to_ros(extracted_inliers)
#    ros_table_msg = pcl_to_ros(extracted_outliers)
#    ros_cluster_msg = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
#    pcl_objects_pub.publish(ros_objects_msg)
#    pcl_table_pub.publish(ros_table_msg)
#    pcl_cluster_pub.publish(ros_cluster_msg)

# Exercise-3 TODOs:
	
    # Classify the clusters! (loop through each detected cluster one at a time)
        # Grab the points for the cluster
    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = out_filtered.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
#        filename = "pcl_cluster" + str(index) + ".pcd"
#        pcl.save(pcl_cluster, filename)

        # Compute the associated feature vector
#        chists = compute_color_histograms(pcl_msg, using_hsv=True)
#        normals = get_normals(pcl_msg)
        
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
#        labeled_features.append([feature, model_name])
#        print("Get feature vector")

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
#        print("Make prediction")

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
#        print("publish RViz label")

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
#        print("Append to detected objects list")

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
    labels = []
    centroids = []

    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    dict_list = []
    yaml_filename = 'output_1.yaml'
    test_scene_num.data = 1

#    print("Detected Object List")
#    print(object_list)

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    print(object_list_param)

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for obj in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        labels.append(obj.label)
        points_arr = ros_to_pcl(obj.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

#    print("Centroids")
#    print(centroids)

    # TODO: Parse parameters into individual variables
    for cnt in range(0, len(object_list_param)):
        object_name.data = object_list_param[cnt]['name']
        object_group = object_list_param[cnt]['group']
        for j in range(0,len(labels)):
            print("Object Name: " + object_name.data + " Label: " + labels[j])
            if object_name.data == labels[j]:
                print("Label found")
                print(labels[j])
                pick_pose.position.x = np.asscalar(centroids[j][0])
                pick_pose.position.y = np.asscalar(centroids[j][1])
                pick_pose.position.z = np.asscalar(centroids[j][2])

        # TODO: Create 'place_pose' for the object
                for j in range(0, len(dropbox_param)):
                    if object_group == dropbox_param[j]['group']:
                        place_pose.position.x = dropbox_param[j]['position'][0]
                        place_pose.position.y = dropbox_param[j]['position'][1]
                        place_pose.position.z = dropbox_param[j]['position'][2]

        # TODO: Assign the arm to be used for pick_place
                if object_group =='green':
                    arm_name.data = 'right'
                elif object_group == 'red':
                    arm_name.data = 'left'

#                print("Details")
#                print(pick_pose)
#                print(place_pose)
#                print(arm_name)

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)

#                print("Dict List")
#                print(dict_list)

        # Wait for 'pick_place_routine' service to come up
#        rospy.wait_for_service('pick_place_routine')

#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
#            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

#            print ("Response: ",resp.success)

#        except rospy.ServiceException, e:
#            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
#    print("Dict List")
#    print(dict_list)
    send_to_yaml(yaml_filename, dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
#    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
#    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
#    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
#    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
