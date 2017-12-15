#pragma once
#ifndef PCV_TEST2_PANORAMIC_STITCHING_H_
#define PCV_TEST2_PANORAMIC_STITCHING_H_
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include <math.h>
//extern "C" {
//#include <generic.h>
//#include <stringop.h>
//#include <pgm.h>
//#include <sift.h>
//#include <getopt_long.h>
//};



namespace Panor {

	class CPanoramic
	{
	public:
	typedef struct {
			int k1;
			int k2;
			double score;
		} Pair;
	//std::vector<CPanoramic::Pair>compare(std::vector<float*>descr1, std::vector<float*>descr2, int K1, int K2, int ND, float thresh);

	public:
		CPanoramic(std::string path, std::string format);
		~CPanoramic(void);

		bool ReadImg();
		/*Detecting matching points and generating descriptors with SIFT, 
		then output the good-match pairs of points in Mat format. */
		void CalKP(int num1,int num2);
		/*Found random 4 points and store those random number in vector—idx*/
		void Found(int iter_num);
		/*In corresponding matching points to compute a homography by DLT (First normalized).
		Using homography to compute the outlier proportion—e, 
		then using e to compute the N which is the iteration times then do sample_count ++ , 
		finally terminate when N > sample_count.*/
		cv::Mat normalize(std::vector<cv::Point> found_pair,int model);
		cv::Mat DLT(int iter_num);
		float take_or_not(cv::Mat each_h);
		cv::Mat runRansac();
		cv::Mat image_transform(int num_one, int num_two);
		void Linear_Blending();
		void test();

	public:
		std::vector<std::string>filelist;
		std::vector<cv::Mat> data;
		std::vector<cv::DMatch> matches;
		std::vector < cv::KeyPoint > keypoints1, keypoints2;
		std::vector< cv::DMatch > good_matches;
		std::vector<cv::Point> match_points;
		std::vector<int> idx;
		std::vector<cv::Point> pair1;
		std::vector<cv::Point> pair2;
		int modelpoints ;
	};
}
#endif