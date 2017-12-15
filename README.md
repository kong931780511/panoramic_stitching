# panoramic_stitching
panoramic_stitching

##Algorithm Principle
>* Detecting matching points and generating descriptors with SIFT, then output the good-match pairs of points in Mat format.  
>* Found random 4 points and store those random number in vector—idx 
>* In corresponding matching points to compute a homography by DLT. Using homography to compute the outlier proportion—e, then using e to compute the N which is the iteration times then do sample_count ++ , finally terminate when N > sample_count. 
>* Taking one of the image as the standard coordinate and computing the homography for all other images which transform their coordinate to standard coordinate. 
>* Compute the maximum value and minimum value of X and Y and using that to determine the final panorama Mat’s size.  
>* For every pixel in panorama (initialize as a black Mat), take the maximum value in every position among all the images as the corresponding pixel value in the panorama Mat. 
