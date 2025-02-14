#######################################################
# Author: Amit Pratap                                                   #
# 
#   this contains the code for Part1, question 1-3
#######################################################

import numpy as np
import cv2
import os
import yaml


class VisualOdometry:
    def __init__(self,k, dist_threshold = 0.6):        
        # Initialize variables to store current and previous frames
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

        self.dist_threshold = dist_threshold 

        self.sift = cv2.SIFT_create()

        # The camera calibration matrix
        self.k = k

    def main(self,frame):
        """
        This is the main function that returns the rotational and the translational vector between frames.

        returns:
            resp: Success or not
            r: rotational matrix of thr lastest frame wrt the previous
            t: translational matrix of the latest frame wrt the previous
        """
        if self.prev_frame is None:
            self.prev_frame = frame  #or frame.copy() if the frames are cv2 object
            self.prev_kp, self.prev_des = self.extract_features(frame)
            return 'First Frame. Provide atleat two!!!', None, None
        else:
            kp, des = self.extract_features(frame)
            match = self.match_features(self.prev_des, des)
            r, t = self.estimate_EssentialMatrix(match, self.prev_kp, kp)
            return 'Success', r, t
            
    def extract_features(self,image):
        """
        This helps the extraction of features from the image and returns keypoint despcriptor pair
        """
        kp, des = self.sift.detectAndCompute(image, None)
        
        return kp, des 

    def match_features(self,des1, des2):
        """
        This matches the two latest descriptor pairs, sets a filter distance for them and reurns 
        the filtered features for further processing
        """

        # # Initiate SIFT detector
        # sift = cv2.SIFT()

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        match = bf.knnMatch(des1,des2, k=2)

        filtered_match = self.filter_matches_distance(match)

        return filtered_match
    
    def filter_matches_distance(self,match):
        """
        This function filters the matches according to a specific threshold and returns 
        those matches to prevent matches with large distances
        """
        
        filtered_match = []
        
        for m,n in match:
            if m.distance < self.dist_threshold*n.distance:
                filtered_match.append([m])

        return filtered_match
    
    def estimate_EssentialMatrix(self, match, kp1, kp2):
        """
        This function estimates the essential matrix and decomposes it to return the 
        rotational and translational vector.
        """
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))

        image1_points = np.float32([kp1[m[0].queryIdx].pt for m in match])
        image2_points = np.float32([kp2[m[0].trainIdx].pt for m in match])

        
        E = cv2.findEssentialMat(image1_points, image2_points, self.k)[0]
        _, rmat, tvec, _ = cv2.recoverPose(E, image1_points, image2_points, self.k)
        
        return rmat, tvec
    
if __name__=="__main__":

    # load YAML file
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)  

    # Camera matrix(given)
    K = np.array(data.get("CAMERA_MATRIX",[[640.,0.,640.], [0.,480.,480.],[0.,0.,1.]]))
    VO = VisualOdometry(K)

    # get the relative path folder name that contains the rgb data
    # please make sure the data png is of the format 00000{n}.png where n is the nth frame
    data_folder_name = data.get("DATA_FOLDER_NAME",'./data/')

    # list the directories according to their number for sequential camera VO
    dirs = [int(f.split('.')[0]) for f in os.listdir(data_folder_name)]
    dirs.sort()

    # print(dirs)
    for dir in dirs:

        # to read the frames correctlt
        if dir == 0:
            n = 0
        else:
            n = int(np.log(dir)/np.log(10))
        # load frame 
        frame1 = cv2.imread(data_folder_name + (5-n)*'0'+f'{dir}.png')

        resp, r, t = VO.main(frame1)

        if resp != 'Success':
            print(resp)
        else:
            print("-"*100)
            print(f"For frame {dir-1} and {dir}")
            print(f"Rotation: {r}")
            print(f"Translation: {t}")
            print("-"*100)
    
    
