# panoramic_stitching
panoramic_stitching

## Algorithm Principle
>* Detecting matching points and generating descriptors with SIFT, then output the good-match pairs of points in Mat format.  
>* Found random 4 points and store those random number in vector—`idx`
>* In corresponding matching points to compute a homography by DLT. Using homography to compute the outlier proportion—`e`, then using `e` to compute the `N` which is the iteration times then do `sample_count ++` , finally terminate when `N` > `sample_count`. 
>* Taking one of the image as the standard coordinate and computing the homography for all other images which transform their coordinate to standard coordinate. 
>* Compute the maximum value and minimum value of `X` and `Y` and using that to determine the final panorama Mat’s size.  
>* For every pixel in panorama (initialize as a black Mat), take the maximum value in every position among all the images as the corresponding pixel value in the panorama Mat. 

## Result
### mountain panorama with 3 pics
![](https://github.com/kong931780511/panoramic_stitching/raw/master/data/result1.png)
![](https://github.com/kong931780511/panoramic_stitching/raw/master/data/result2.png)
### church panorama with 8 pics
![](https://github.com/kong931780511/panoramic_stitching/raw/master/data/result3.png)
### houses panorana with 8 pics
![](https://github.com/kong931780511/panoramic_stitching/raw/master/data/result4.png)

## Conclusion
 
>As shown in the result section, the panorama is somewhat skew. Under several attempts, I got the conclusion that the panorama image’s quality is largely depend on the image which is treated as the standard one (Compared between figure1 and figure2). I also using average of every images’ pixel value to compute the panorama, but I got a pretty blurred image so I chose to use the max value and got a relatively better result.  
