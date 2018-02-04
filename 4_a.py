import cv2 # opencv-python

image = cv2.imread('baby.jpg') # read target image
target = cv2.imread('zespri.jpg')         # read image

matching_method = input('Matching method (1.Brute-Force Matching 2.FLANN based Matcher): ') # input the mathcing method

sift = cv2.xfeatures2d.SIFT_create() # initiate SIFT detector

# find the key point and descriptors for image and target image by used "sift.detectAndCompute"
img_keypoint, img_des = sift.detectAndCompute(image,None)  # get key points and descriptors for image
tar_keypoint, tar_des = sift.detectAndCompute(target,None) # get key points and descriptors for target image

### Data Structure for SIFT matches => List [ [,][,][,][,][,][,][,][,][,][,]....]

if (matching_method == 1): # if input = 1,  Brute-Force Matching
    # find the match key points by using opencv function "cv2.BFMatcher"
    match_KeyPoint = cv2.BFMatcher()
    matches = match_KeyPoint.knnMatch(tar_des,img_des, k=2) # Match descriptors
    
### another drawing method--------------------------------------------------------------###
    """
    # draw key points for two image
    image_2 = cv2.drawKeypoints(image, img_keypoint, None, color=(0,0,255), flags=0)
    target_2 = cv2.drawKeypoints(target, tar_keypoint, None, color=(0,0,255), flags=0)
       
    good_BF = [] # create an empty object for record the good matches depend on the distance
    
    # apply ratio test 
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_BF.append([m])
            
    # draw matches key points by using "cv2.drawMatchesKnn"
    result_img = cv2.drawMatchesKnn(target_2,tar_keypoint,image_2,img_keypoint,good_BF,None, flags=2)
    """
###-------------------------------------------------------------------------------------###
    # create an empty object for record the good matches depend on the distance
    good_BF = [[0,0] for i in xrange(len(matches))]
    
    # apply ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_BF[i]=[1,0]   
    
    # set parameter for drawing        
    draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255), matchesMask = good_BF, flags = 0)
    
    # draw matches key points by using "cv2.drawMatchesKnn"
    result_img = cv2.drawMatchesKnn(target,tar_keypoint,image,img_keypoint,matches,None,**draw_params)       
    
    #cv2.imwrite('4_a_BF.jpg', result_img) # output result image name 4_a.jpg           
    cv2.imshow('4_a.jpg', result_img) # show result image
    cv2.waitKey(0) # system pause 

if (matching_method == 2): # if input = 2, FLANN based Matcher
    
    ### FLANN parameters
    ## Constructs a nearest neighbor search index for a given dataset
    index_params = dict(algorithm = 0, trees = 5) # FLANN_INDEX_KDTREE = 0
    ## Performs a K-nearest neighbor search for a given query point using the index
    search_params = dict(checks=50) # Higher checks value gives better precision
    
    # find matches by used "cv2.FlannBasedMatcher" and "knnMatch"
    matches = cv2.FlannBasedMatcher(index_params,search_params).knnMatch(tar_des,img_des,k=2)
    #-----------------------------------------------------------------------------------------------------#
    # public:                                                                                             #  
    #    FlannBasedMatcher                                                                                #
    #      const Ptr<flann::IndexParams>& indexParams=new flann::KDTreeIndexParams(),                     #
    #      const Ptr<flann::SearchParams>& searchParams=new flann::SearchParams() );                      #
    # ----------------------------------------------------------------------------------------------------#
    # C++: void DescriptorMatcher::knnMatch(InputArray queryDescriptors, InputArray trainDescriptors,     #
    # vector<vector<DMatch>>& matches, int k, InputArray mask=noArray(), bool compactResult=false ) const #
    #-----------------------------------------------------------------------------------------------------#


### another drawing method------------------------------------------------------------------------###
    """
    # draw key points for two image
    image_2 = cv2.drawKeypoints(image, img_keypoint, None, color=(0,0,255), flags=0)
    target_2 = cv2.drawKeypoints(target, tar_keypoint, None, color=(0,0,255), flags=0)
    
    good_FLANN = [] # create an empty object for record the good matches depend on the distance
    
    # apply ratio test 
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_FLANN.append([m])
            
    result_img = cv2.drawMatchesKnn(target_2,tar_keypoint,image_2,img_keypoint,good_FLANN,None, flags=2)   
    """     
###-----------------------------------------------------------------------------------------------###        
    # create an empty object for record the good matches depend on the distance
    good_FLANN = [[0,0] for i in xrange(len(matches))]
    
    # apply ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good_FLANN[i]=[1,0]
    
    # set parameter for drawing
    draw_params = dict(matchColor = (0,255,0), singlePointColor = (0,0,255),
                       matchesMask = good_FLANN, flags = 0)
    
    # draw matches by using "cv2.drawMatchesKnn"
    result_img = cv2.drawMatchesKnn(target,tar_keypoint,image,img_keypoint,matches,None,**draw_params)
    
    #cv2.imwrite('4_a_FLANN.jpg', result_img) # output result image name 4_a.jpg           
    cv2.imshow('4_a', result_img) # show result image
    cv2.waitKey(0) # system pause 








