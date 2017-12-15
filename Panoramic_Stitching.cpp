#include "Panoramic_Stitching.h"

namespace Panor 
{
	CPanoramic::CPanoramic(std::string path, std::string format)
	{
		cv::Directory dir;
		filelist = dir.GetListFiles(path, format, true);
		std::cout << filelist[2] << std::endl;
		std::vector<cv::Point> match_points;
		modelpoints = 4;
	}

	CPanoramic::~CPanoramic()
	{
		 
	}

	bool CPanoramic::ReadImg()
	{ 
		if (filelist.empty())
			return false;
		//read image
		for (int i = 0; i < filelist.size(); i++)
		{
			cv::Mat pic = cv::imread(filelist[i]);
			data.push_back(pic);
		}
		if (data.empty())
			return false;
		else
			return true;
	}

	void CPanoramic::CalKP(int num1,int num2)
	{
		match_points.clear();
		pair1.clear();
		pair2.clear();
		matches.clear();
		keypoints1.clear();
		keypoints2.clear();
		good_matches.clear();
		cv::resize(data[num1],data[num1],cv::Size(800,600));
		cv::resize(data[num2], data[num2], cv::Size(800, 600));

		cv::SiftFeatureDetector detector;
		detector.detect(data[num1], keypoints1);
		detector.detect(data[num2], keypoints2);
		cv::SiftDescriptorExtractor extractor;
		cv::Mat descriptor1, descriptor2;
		extractor.compute(data[num1], keypoints1, descriptor1);
		extractor.compute(data[num2], keypoints2, descriptor2);
		cv::FlannBasedMatcher matcher;
		matcher.match(descriptor1, descriptor2, matches);
		cv::Mat outimg;
		cv::drawMatches(data[num1], keypoints1, data[num2], keypoints2, matches, outimg);
		double max_dist = 0; double min_dist = 100;
		for (int i = 0; i < descriptor1.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		for (int i = 0; i < descriptor1.rows; i++)
		{
			if (matches[i].distance <= cv::max(2 * min_dist, 0.02))
			{
				good_matches.push_back(matches[i]);
			}
		}

		for (int i = 0; i < good_matches.size(); i++)
		{
			cv::Point p1 = cv::Point(keypoints1[good_matches[i].queryIdx].pt);
			cv::Point p2 = cv::Point(keypoints2[good_matches[i].trainIdx].pt);
			match_points.push_back(p1);
			match_points.push_back(p2);
		}

			for (int i = 0; i < match_points.size() / 2; i++)
			{
				pair1.push_back(match_points[i * 2]);
				pair2.push_back(match_points[i * 2 + 1]);
			}
	}

	void CPanoramic::Found(int iter_num)
	{

		int idx_i = 0;
		int iters = 0;
		int j = 0;
		int i = 0;
		CvRNG rng = cvRNG(cvGetTickCount());
		for (; iters < iter_num; iters++)
		{
			for (i=0; i < modelpoints; i++)
			{
				idx_i = cvRandInt(&rng) % (pair1.size());
				idx.push_back(idx_i);
				for ( j = 0; j < i; j++)
				{
					if (idx[iters*modelpoints+j] == idx_i)///////////////check if repeat
					{
						idx.pop_back();
						i -= 1;
						break;
					}
				}
			}
		}
	}
	/////////////////choice==0--->p1,choice==1---->p2
	cv::Mat CPanoramic::normalize(std::vector<cv::Point> found_pair,int model)
	{
		
		int sumx = 0;
		int sumy = 0;
		for (int i = 0; i < found_pair.size(); i++)
		{
			sumx += found_pair[i].x;
			sumy += found_pair[i].y;
		}
		int sumx_mean = sumx / found_pair.size();
		int sumy_mean = sumy / found_pair.size();
		int l_sumx = 0;
		int l_sumy = 0;
		for (int i = 0; i < found_pair.size(); i++)
		{
			l_sumx += fabs(found_pair[i].x - sumx_mean);
			l_sumy += fabs(found_pair[i].y - sumy_mean);
		}
		cv::Mat L = cv::Mat::zeros(3, 3, CV_32F);
		L.at<float>(0, 0) = (float)found_pair.size() / l_sumx;
		L.at<float>(0, 2) = (float)-sumx_mean*found_pair.size() / l_sumx;
		L.at<float>(1, 1) = (float)found_pair.size() / l_sumy;
		L.at<float>(1, 2) = (float)-sumy_mean*found_pair.size() / l_sumy;
		L.at<float>(2, 2) = (float)1;
		cv::Mat ori = cv::Mat(3, found_pair.size(), CV_32F);
		cv::Mat trans = cv::Mat(3, found_pair.size(), CV_32F);
		for (int i = 0; i < found_pair.size(); i++)
		{
			ori.at<float>(0, i) = found_pair[i].x;
			ori.at<float>(1, i) = found_pair[i].y;
			ori.at<float>(2, i) = 1;
		}
		trans = L*ori;
		if (model == 1)
			return trans;
		else
			return L;
	}

	cv::Mat CPanoramic::DLT(int iter_num)
	{
		cv::Mat pair1_found = cv::Mat::ones(3, modelpoints, CV_32F);
		cv::Mat pair2_found = cv::Mat::ones(3, modelpoints, CV_32F);
		cv::Mat pair_norm1 = normalize(pair1, 1);
		cv::Mat pair_norm2 = normalize(pair2, 1);
		cv::Mat A = cv::Mat::zeros(3 * modelpoints, 9, CV_32F);
		for (int i = 0; i < modelpoints; i++)
		{
			A.at<float>(i * 3, 3) = (float)-pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3, 4) = (float)-pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3, 5) = (float)-pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3, 6) = pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3, 7) = pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3, 8) = pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 0) = pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 1) = pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 2) = pair_norm1.at<float>(2, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 6) = (float)-pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 7) = (float)-pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 1, 8) = (float)-pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 0) = (float)-pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 1) = (float)-pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 2) = (float)-pair_norm1.at<float>(1, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 3) = pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(0, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 4) = pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(1, idx[iter_num*modelpoints + i]);
			A.at<float>(i * 3 + 2, 5) = pair_norm1.at<float>(0, idx[iter_num*modelpoints + i])*pair_norm2.at<float>(2, idx[iter_num*modelpoints + i]);

		}
		cv::Mat w,u,v;
		cv::SVDecomp(A, w, u, v);
	    cv::Mat h_ori(v.row(8));
		cv::Mat h = cv::Mat(3, 3, CV_32F);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				h.at<float>(i, j) = h_ori.at<float>(i * 3 + j);
			}
		}

		for (int j = 0; j < modelpoints; j++)
		{
			pair1_found.at<float>(0, j) = pair_norm1.at<float>(0, idx[iter_num*modelpoints + j]);
			pair1_found.at<float>(1, j) = pair_norm1.at<float>(1, idx[iter_num*modelpoints + j]);
			pair2_found.at<float>(0, j) = pair_norm2.at<float>(0, idx[iter_num*modelpoints + j]);
			pair2_found.at<float>(1, j) = pair_norm2.at<float>(1, idx[iter_num*modelpoints + j]);
		}
		cv::Mat pair1_T = normalize(pair1, 0);
		cv::Mat pair2_T = normalize(pair2, 0);
		cv::Mat pair1_T_inv;
		cv::invert(pair1_T, pair1_T_inv);
		cv::Mat H_each = pair1_T_inv*h*pair2_T;
		return H_each;
	}

	float CPanoramic::take_or_not(cv::Mat each_h)
	{
		cv::Mat pair1_each = cv::Mat::ones(3,pair1.size(),CV_32F);
		cv::Mat pair2_each = cv::Mat::ones(3, pair2.size(), CV_32F);
		for (int i = 0; i < pair1.size(); i++)
		{
			pair1_each.at<float>(0, i) = pair1[i].x;
			pair1_each.at<float>(1, i) = pair1[i].y;
			pair1_each.at<float>(2, i) = 1;
			pair2_each.at<float>(0, i) = pair2[i].x;
			pair2_each.at<float>(1, i) = pair2[i].y;
			pair2_each.at<float>(2, i) = 1;
		}
		cv::Mat pair_trans = each_h*pair2_each;
		cv::Mat pair_trans_rc= cv::Mat::ones(3, pair1.size(), CV_32F);
		for (int i = 0; i < pair1.size(); i++)
		{ 
			pair_trans_rc.at<float>(0, i) = pair_trans.at<float>(0, i) / pair_trans.at<float>(2, i);
			pair_trans_rc.at<float>(1, i) = pair_trans.at<float>(1, i) / pair_trans.at<float>(2, i);
			pair_trans_rc.at<float>(2, i) = 1;
		}
		cv::Mat pair_delta = (pair_trans_rc - pair1_each);
		pair_delta = pair_delta.mul(pair_delta);
		cv::Mat Pair_distance(pair_delta.row(0) + pair_delta.row(1));
		int count = 0;
		for (int i = 0; i < Pair_distance.cols*Pair_distance.rows; i++)
		{
			//std::cout << Pair_distance.at<float>(0, i) << std::endl;
			if (Pair_distance.at<float>(0, i) > 9)
			{ 
				//std::cout << i << std::endl;
				count++;
			}
		}
		float e = (float)count / (float)(pair1.size());
		std::cout <<"e_ori: "<< count<< std::endl;
		return e;
	}

	cv::Mat CPanoramic::runRansac()
	{
		cv::Mat H_final = cv::Mat(3,3,CV_32F);
		int N_init = 2000;
		int sample_count = 0;
		float confidence = 0.95;
		float turn = 0;
		int N = N_init;
		cv::Mat H_DLT(3, 3, CV_32F);
		idx.clear();
		Found(N);
		while (N > sample_count)
		{
		H_DLT = DLT(sample_count);
				 turn=take_or_not(H_DLT);
				 std::cout <<"e: "<< turn << std::endl;
				 float q = pow((1 - turn), modelpoints);
				 N = log(1-confidence)/log(1-q);
				 std::cout << "N: "<<N << std::endl;
				 std::cout << "sample_count: " << sample_count << std::endl;
				 sample_count++;
			}
		//	std::cout <<"sample_count: "<< sample_count << std::endl;
		H_final = H_DLT;
			/////////calculate homography
	
			std::cout <<"H:"<< H_final << std::endl;
			return H_final;
	}



	cv::Mat CPanoramic::image_transform(int num_one, int num_two)
	{
		 CalKP(num_one,num_two);
		 cv::Mat final_image_each= runRansac();
		 return final_image_each;
	}

	void CPanoramic::Linear_Blending()
	{
		std::vector<cv::Mat> H_total;
		float tem_min_x = 0;
		float tem_min_y = 0;
		float tem_max_x = 0;
		float tem_max_y = 0;
		std::vector<cv::Mat> H_between;
		for (int i = 0; i < data.size() - 1; i++)
		{
			cv::Mat HH = image_transform(i, i + 1);
			H_between.push_back(HH);
		}
			std::cout <<"3reslut:"<<H_between[data.size()-2] << std::endl;
/*Taking one of the image as the standard coordinate and computing the homography
for all other images which transform their coordinate to standard coordinate.*/
		for (int i = 0; i < data.size()-1; i++)
		{
			if (i == 0)
			{
				H_total.push_back(H_between[i]);
			}
			else
			{
				//std::cout << "new H: " << HH << std::endl;
				H_total.push_back(H_total[i - 1] * H_between[i]);
			}
		}
/*. Compute the maximum value and minimum value of X and Y and using that to determine the final panorama Mat’s size. */
			for (int i = 0; i < data.size()-1; i++)
			{
				for (int j = 0; j < data[0].cols; j+=data[0].cols-1)
				{
					for (int m = 0; m < data[0].rows; m+=data[0].rows-1)
					 {
						float x = j*H_total[i].at<float>(0, 0) + m*H_total[i].at<float>(0, 1) + H_total[i].at<float>(0, 2);
						float y = j*H_total[i].at<float>(1, 0) + m*H_total[i].at<float>(1, 1) + H_total[i].at<float>(1, 2);
						float z = j*H_total[i].at<float>(2, 0) + m*H_total[i].at<float>(2, 1) + H_total[i].at<float>(2, 2);
						x = x / z;
						y = y / z;
						if (x < tem_min_x)
							tem_min_x = x;
						if (y < tem_min_y)
							tem_min_y = y;
						if (x > tem_max_x) {
							tem_max_x = x;
						}
						if (y > tem_max_y)
							tem_max_y = y;
					}
				}
			}
		std::cout << tem_min_x << " " << tem_min_y << " " << tem_max_x << " " << tem_max_y << std::endl;
		cv::Mat H_move = cv::Mat(3, 3, CV_32F);
		cv::Mat img_panorama = cv::Mat::zeros((tem_max_y + fabs(tem_min_y)), (tem_max_x + fabs(tem_min_x)), CV_8UC3);
		H_move.at<float>(0, 0) = 1;
		H_move.at<float>(0, 1) = 0;
		H_move.at<float>(0, 2) = fabs(tem_min_x);
		H_move.at<float>(1, 0) = 0;
		H_move.at<float>(1, 1) = 1;
		H_move.at<float>(1, 2) = fabs(tem_min_y);
		H_move.at<float>(2, 0) = 0;
		H_move.at<float>(2, 1) = 0;
		H_move.at<float>(2, 2) = 1;
		for (int i = 0; i < data.size()-1; i++)
		{
			H_total[i] = H_move*H_total[i];
			cv::warpPerspective(data[i + 1], data[i + 1], H_total[i], cv::Size(img_panorama.cols,img_panorama.rows), CV_INTER_LINEAR);
			//cv::imshow("33", data[i + 1]);
			//cv::waitKey(0);
		}
		cv::Mat H_I= cv::Mat::zeros(3, 3, CV_32F);
		for (int i = 0; i < 3; i++)
		{
			H_I.at<float>(i, i) = 1;
		}
		cv::warpPerspective(data[0], data[0], H_move*H_I, cv::Size(img_panorama.cols,img_panorama.rows), CV_INTER_LINEAR,0,cv::Scalar(0,0,0));
		
		/*. For every pixel in panorama (initialize as a black Mat), 
		take the maximum value in every position among all the images
		as the corresponding pixel value in the panorama Mat.*/
		for (int i = 0; i < img_panorama.rows; i++)
		{
			for (int j = 0; j < img_panorama.cols; j++)
			{
				int sumR = 0;
				int sumG = 0;
				int sumB = 0;
				int count = 0;
				for (int m = 0; m < data.size(); m++)
				{
					if (data[m].at<cv::Vec3b>(i, j)[0] != 0 || data[m].at<cv::Vec3b>(i, j)[1] != 0 || data[m].at<cv::Vec3b>(i, j)[2] != 0)
					{
						if (data[m].at<cv::Vec3b>(i, j)[0] > img_panorama.at<cv::Vec3b>(i, j)[0])
							img_panorama.at<cv::Vec3b>(i, j)[0] = data[m].at<cv::Vec3b>(i, j)[0];
						if (data[m].at<cv::Vec3b>(i, j)[1] > img_panorama.at<cv::Vec3b>(i, j)[1])
							img_panorama.at<cv::Vec3b>(i, j)[1] = data[m].at<cv::Vec3b>(i, j)[1];
						if (data[m].at<cv::Vec3b>(i, j)[2] > img_panorama.at<cv::Vec3b>(i, j)[2])
							img_panorama.at<cv::Vec3b>(i, j)[2] = data[m].at<cv::Vec3b>(i, j)[2];
					}
				}
			}
		}
		cv::imwrite("panor.jpg", img_panorama);
		imshow("panor", img_panorama);
		cv::waitKey(0);

	}

	void CPanoramic::test()
	{

			Linear_Blending();

	}
}