import numpy as np
import cv2
from scipy.optimize import least_squares
import open3d

def videoToArray(video_path):
    frames = []
    video = cv2.VideoCapture(video_path) #open video
    success = True
    while success: #get all frames from video 
        success,image = video.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            frames.append(image) #add image to list
            
    frames = np.array(frames) #convert to np array
    return frames

def readDinoRingImages():
    frames = []
    K = np.array([[3310.400000, 0.000000, 316.730000], [0.000000, 3325.500000, 200.550000],[0.000000, 0.000000, 1.000000]])
    R = np.array([[-0.08661715685291285200, 0.97203145042392447000, 0.21829465483805316000], [-0.97597093004059532000, -0.03881511324024737600, -0.21441803766270939000], [-0.19994795321325870000, -0.23162091782017200000, 0.95203517502501223000]])
    t = np.array([[-0.0526034704197], [0.023290917003], [0.659119498846]])
    name = "dinoRing"
    n = 48
    for i in range(n):
        img = cv2.imread('data/dinoRing/dinoR0{}.png'.format(f'{i+1:03}'))
        frames.append(img)

    frames = np.array(frames) #convert to np array
    return frames, K, name

def readTempleRingImages():
    K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
    R = np.array([[0.02187598221295043000, 0.98329680886213122000, -0.18068986436368856000], [0.99856708067455469000, -0.01266114646423925600, 0.05199500709979997700], [0.04883878372068499500, -0.18156839221560722000, -0.98216479887691122000]])
    t = np.array([[-0.0292149526928], [-0.0241923869131], [0.52269561933]])
    name = "templeRing"
    frames = []
    n = 12
    for i in range(n):
        img = cv2.imread('data/templeRing/templeR0{}.png'.format(f'{i+1:03}'))
        frames.append(img)

    frames = np.array(frames) #convert to np array
    return frames, K, name 
    
def calculateDisparity(img1, img2):
    #window_size = 3
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=7,
        P1=8,
        P2=32,
        disp12MaxDiff=2,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=63)
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(img1, img2)  
    dispr = right_matcher.compute(img2, img1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filtered_img = wls_filter.filter(displ, img1, None, dispr)
    disparity = cv2.normalize(filtered_img, filtered_img, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX) #normalize to 0-255
    return disparity

def featureMatch(img1, img2, threshold = .7):
    #create sift feature detector
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    #match using brute force approach
    bf_match = cv2.BFMatcher()
    matches = bf_match.knnMatch(des1, des2, k=2)

    #ratio test
    good_feature = []
    for m,n in matches:
        if m.distance < threshold * n.distance:
            good_feature.append(m)

    #return keypoints for img1 and img2
    np.float32([kp1[m.queryIdx].pt for m in good_feature])
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_feature])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_feature])
    
    return pts1, pts2

def triangulate3DPoints(P1,P2,pts1,pts2):
    # Return 3D points for each of the 2D image pairs
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_4D / points_4D[3] 
    points_3D = points_3D[:3,:]
    return points_3D

def reprojectionError(points_3D, image_points, transform_matrix, K):
        # calculate error between 2D image points and reprojected 2D image points from 3D points
        R = transform_matrix[:3, :3]
        T = transform_matrix[:3, 3]
        r_vec, _ = cv2.Rodrigues(R)
        
        image_points_calc, _ = cv2.projectPoints(points_3D, r_vec, T, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        #print(image_points_calc.shape)
        total_error = cv2.norm(image_points_calc, np.float32(image_points), cv2.NORM_L2)
        
        return total_error / len(image_points_calc)

def optimal_reprojection_error(obj_points):
        '''
        calculates of the reprojection error during bundle adjustment
        returns error 
        '''
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

def bundle_adjustment(point_3D, opt, transform_matrix_new, K, r_error):
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, point_3D.ravel()))

        values_corrected = least_squares(optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape(3,(int(len(values_corrected[21 + rest:])/3))), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))
    
def findCommonPoints(pts1, pts1_2, pts2):
    #pts1 = feature matches in img1 between img0 and img 1
    #pts1_2 = feature matches in img1 between img1 and img 2
    #pts2 = feature matches in img2 between img1 and img 2
    print(pts1.shape, pts1_2.shape, pts2.shape)
    cm_pts_01 = []
    cm_pts_12 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts1_2 == pts1[i, :])
        if a[0].size != 0:
            cm_pts_01.append(i)
            cm_pts_12.append(a[0][0])

    mask_array_1 = np.ma.array(pts1_2, mask=False)
    mask_array_1.mask[cm_pts_12] = True
    mask_array_1 = mask_array_1.compressed()
    mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

    mask_array_2 = np.ma.array(pts2, mask=False)
    mask_array_2.mask[cm_pts_12] = True
    mask_array_2 = mask_array_2.compressed()
    mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
    print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
    
    return np.array(cm_pts_01), np.array(cm_pts_12), mask_array_1, mask_array_2

def PnP(points_3D, pts1, K, dist, pts2):
    #use PnP algorithm
    print(points_3D.shape)
    print(pts1.shape)
    _, r_vec, t, mask = cv2.solvePnPRansac(points_3D, pts1, K, dist, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(r_vec)
    
    if mask is not None:
        pts1 = pts1[mask[:, 0]]
        points_3D = points_3D[mask[:, 0]]
        pts2 = pts2[mask[:, 0]]
    return R, t, pts1, points_3D, pts2

def to_ply(path, name, point_cloud, colors):
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        #print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '\\res\\' + name +'_SFM.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
            
def write_ply(path, name, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.reshape(-1, 3)
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(path + '\\res\\' + name +'_MVS.ply', 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def normalizePoints(points):
    n = len(points.T)
    points_avg = points - np.sum(points,axis=0)/n
    #print(np.sum(points_avg,axis=0))
    return points_avg