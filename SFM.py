import numpy as np
import cv2
from matplotlib import pyplot as plt
import helper_functions as hf
import os
import open3d as o3d

print("running...")

#compile data from images
frames_color, K, name = hf.readDinoRingImages()
frames = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_color])

#initialize needed variables
total_images, h, w = frames.shape
T0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
T1 = np.empty((3, 4))
P0 = np.matmul(K, T0)
P1 = np.empty((3, 4))
points = np.zeros((3, 1))
colors = np.zeros((3, 1))

# use SIFT feature extracter, brute force matcher, remove weak matches
img0 = frames[0]
img1 = frames[1]
pts0, pts1 = hf.featureMatch(img0, img1)
#print(len(pts0))

#get E matrices
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.FM_RANSAC, prob=0.999, threshold=.4)
pts0 = pts0[mask.ravel() == 1]
pts1 = pts1[mask.ravel() == 1]

# Decompose E using SVD 
_,R,t,e_mask = cv2.recoverPose(E,pts0,pts1,K)
pts0 = pts0[e_mask.ravel() > 0]
pts1 = pts1[e_mask.ravel() > 0]

# Transformation matrices
T1[:3, :3] = np.matmul(R, T0[:3, :3])
T1[:3, 3] = np.matmul(T0[:3, :3], t.ravel()) + T0[:3, 3]
P1 = np.matmul(K,T1)

# Triangulate the 3D points
print("# of points: ", len(pts1))
points_3D = hf.triangulate3DPoints(P0,P1,pts0.T,pts1.T) #3xN matrix of 3D poiints

# Find reprojection error
error = hf.reprojectionError(points_3D, pts1, T1, K)
print("Reprojection Error: ", error)

_, _, pts1, points_3D, _ = hf.PnP(points_3D.T, pts1, K, np.zeros((5, 1), dtype=np.float32), pts0)
points_3D = points_3D.T

for i in range(total_images-2):
    # use SIFT feature extracter, brute force matcher, remove weak matches
    img2 = frames[i+2]
    img2_color = frames_color[i+2]
    print("Images: ", i+1," " ,i+2)
    pts1_2, pts2 = hf.featureMatch(img1, img2) #matches between img1 and img2,

    if i != 0:
        points_3D = hf.triangulate3DPoints(P0,P1,pts0.T,pts1.T)
    
    #find common points
    cm_pts_0, cm_pts_1, cm_mask_0, cm_mask_1 = hf.findCommonPoints(pts1, pts1_2, pts2)
    cm_pts_2 = pts2[cm_pts_1]
    cm_pts_1_2 = pts1_2[cm_pts_1]
    
    #PnP to get R,t
    points_3D = points_3D.T
    R, t, cm_pts_2, points_3D, cm_pts_1_2 = hf.PnP(points_3D[cm_pts_0], cm_pts_2, K, 0, cm_pts_1_2)
    T2 = np.hstack((R, t))
    P2 = np.matmul(K, T2)
    
    error =  hf.reprojectionError(points_3D, cm_pts_2, T2, K)

    if len(pts1) > 0:
        # Triangulate the 3D points
        print("# of points: ", len(cm_mask_0))
        points_3D = hf.triangulate3DPoints(P1,P2,cm_mask_0.T,cm_mask_1.T) #3xN matrix of 3D poiints

        # Find reprojection error
        error = hf.reprojectionError(points_3D, cm_mask_1, T2, K)
        print("Reprojection Error: ", error)
        
        enable_bundle_adjustment = 0
        if enable_bundle_adjustment:
                points_3D, cm_mask_1, T2 = hf.bundle_adjustment(points_3D, cm_mask_1, T2, K, .5)
                #print(points_3D.shape)
                P2 = np.matmul(K, T2)
                error = hf.reprojectionError(points_3D, cm_mask_1, T2, K)
                print("Bundle Adjusted error: ",error)
        
        if error<1:
            points = np.hstack((points,points_3D))
            points_left = np.array(cm_mask_1, dtype=np.int32)
            #print("points size: ",points_left.shape)
            color = np.array([img2_color[c[1], c[0]] for c in points_left])
            #print("color size: ",color.shape)
            colors = np.hstack((colors,color.T))
            
        print(points.shape)
    
    #plot reprojection errors
    plt.scatter(i, error)
    plt.title("Reprojection Error")
    plt.xlabel("Images")
    plt.ylabel("Error (pixels)")
    plt.pause(0.05)
    
    #update matrices for next loop
    T0 = np.copy(T1)
    P0 = np.copy(P1)
    img0 = np.copy(img1)
    img1 = np.copy(img2)
    pts0 = np.copy(pts1_2)
    pts1 = np.copy(pts2)
    P1 = np.copy(P2)
    T1 = np.copy(T2)
plt.pause(10)
print("Final shape: ",points.shape)
points = hf.normalizePoints(points)
print("Making .ply file...")
print(points.shape)
path = os.getcwd()
hf.to_ply(path, name, points.T, colors.T)
print("Complete")     

pcd_path = path + '\\res\\' + name +'_SFM.ply'
pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])
