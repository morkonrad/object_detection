#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace fmatches
{
	struct sift_feat
	{
		float x{ 0 };
		float y{ 0 };
		float sigma{ 0 };
		float orient{ 0 };
		float octave_id{ 0 };
		float scale{ 0 };
		std::vector<float> descriptor{ 0 };
	};

	//! Ipoint structure holds a interest point descriptor
	typedef struct {
		float x;
		float y;
		float scale;
		float orientation;
		int laplacian;
		int clusterIndex;
		float descriptor[64];
	} Ipoint;

	int cv_draw_points(const std::vector<sift_feat>& feats, cv::Mat& cimg)
	{
		int cf = 0;
		for (auto& feat : feats)
		{
			if (feat.sigma != 0)
			{
				cv::circle(cimg, cv::Point(feat.x, feat.y), 2, cv::Scalar(0, 255, 0));
				cf++;
			}
		}
		return cf;
	}

	void cv_draw_lines(const std::vector<sift_feat>& feats, cv::Mat& img)
	{
		cv::Scalar color = CV_RGB(200, 0, 0);
		auto scale = 5.0;
		auto hscale = 0.75;
		uint cf = 0;

		for (auto& feat : feats)
		{
			if (feat.sigma != 0.0)
			{
				auto x = feat.x;
				auto y = feat.y;
				auto scl = feat.scale;
				auto ori = feat.orient;

				auto start_x = std::round(x);
				auto start_y = std::round(y);

				auto len = std::round(scl * scale);
				auto hlen = std::round(scl * hscale);
				auto blen = len - hlen;
				auto end_x = std::round(len * std::cos(ori)) + start_x;
				auto end_y = std::round(len * -std::sin(ori)) + start_y;
				auto h1_x = std::round(blen * std::cos(ori + CV_PI / 18.0)) + start_x;
				auto h1_y = std::round(blen * -std::sin(ori + CV_PI / 18.0)) + start_y;
				auto h2_x = std::round(blen * std::cos(ori - CV_PI / 18.0)) + start_x;
				auto h2_y = std::round(blen * -std::sin(ori - CV_PI / 18.0)) + start_y;

				auto start = cv::Point(start_x, start_y);
				auto end = cv::Point(end_x, end_y);
				auto h1 = cv::Point(h1_x, h1_y);
				auto h2 = cv::Point(h2_x, h2_y);

				cv::line(img, start, end, color, 1, 8, 0);
				cv::line(img, end, h1, color, 1, 8, 0);
				cv::line(img, end, h2, color, 1, 8, 0);

				cf++;
			}
		}
	}

	void cv_Plot_features(const std::vector<sift_feat>& feats, const int frame_id,
		const float& frame_time_ms, const std::string& wname, cv::Mat& cimg)
	{
		int count_features = 0;
		count_features = cv_draw_points(feats, cimg);
		cv_draw_lines(feats, cimg);

		std::stringstream ss, ss0, ss1, ss2;

		ss << "Frame id<" << frame_id << ">";
		ss0 << "FPS: <" << 1 / (frame_time_ms / 1000) << ">";
		ss1 << "Features: <" << count_features << ">";
		ss2 << "Cols,Rows: <" << cimg.cols << "," << cimg.rows << ">";

		auto color = cv::Scalar(200, 20, 255);
		putText(cimg, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1, color);
		putText(cimg, ss0.str(), cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1, color);
		putText(cimg, ss1.str(), cv::Point(10, 90), cv::FONT_HERSHEY_PLAIN, 1, color);
		putText(cimg, ss2.str(), cv::Point(10, 120), cv::FONT_HERSHEY_PLAIN, 1, color);
		cv::imshow(wname.c_str(), cimg);
	}
	
	template<typename T>
	void ConvertToSift_feat(
		const std::vector<T /*SIFT::sfeature*/>& feats,
		const std::vector<float>& orientations,
		const std::vector<float>& descriptors,
		std::vector<sift_feat>& kp_out)
	{
		const auto cnt_kp = feats.size();
		const auto desc_size = descriptors.size() / cnt_kp;

		for (int i = 0; i < cnt_kp; i++)
		{
			sift_feat sf;
			sf.x = feats[i].x;
			sf.y = feats[i].y;
			sf.sigma = feats[i].sigma;
			sf.scale = feats[i].scale_id;
			sf.octave_id = feats[i].octave_id;
			sf.orient = orientations[i];
			for (int d = 0; d < desc_size; d++)
				sf.descriptor.push_back(descriptors[d + i * desc_size]);
			kp_out.push_back(sf);
		}
	}

	void ConvertToCVKeyPoint(const std::vector<sift_feat>& kp_in,
		std::vector<cv::KeyPoint>& kp_out, cv::Mat& desc_out)
	{
		desc_out = cv::Mat(kp_in.size(), 128, CV_32FC1);
		int y = 0;
		for (auto item : kp_in)
		{
			kp_out.push_back(cv::KeyPoint(item.x, item.y, item.sigma, item.orient, item.octave_id));
			for (int x = 0; x < desc_out.cols; x++)
				desc_out.at<float>(cv::Point(x, y)) = item.descriptor[x];
			y++;
		}
	}

	void ConvertToCVKeyPoint(const std::vector<Ipoint>& kp_in,
		std::vector<cv::KeyPoint>& kp_out, cv::Mat& desc_out)
	{
		desc_out = cv::Mat(kp_in.size(),64, CV_32FC1);
		int y = 0;
		for (auto item : kp_in)
		{
			kp_out.push_back(cv::KeyPoint(item.x, item.y, item.scale, item.orientation, item.laplacian));
			for (int x = 0; x < desc_out.cols; x++)
				desc_out.at<float>(cv::Point(x, y)) = item.descriptor[x];
			y++;
		}
	}
	
	#define _RANSAC_
	void Plot_Img_Matches_ocv(
		cv::Mat& img_object, cv::Mat& img_scene,
		const std::vector<sift_feat>& kp_object,
		const std::vector<sift_feat>& kp_scene,
		const bool up_scale=false)
	{
		std::vector < cv::KeyPoint> cv_kp_object, cv_kp_scene;
		cv::Mat cv_desc_object, cv_desc_scene;

		ConvertToCVKeyPoint(kp_object, cv_kp_object, cv_desc_object);
		ConvertToCVKeyPoint(kp_scene, cv_kp_scene, cv_desc_scene);

		cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		matcher.match(cv_desc_object, cv_desc_scene, matches);
		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between key points
		for (int i = 0; i < cv_desc_object.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitrary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		//-- PS.- radiusMatch can also be used here.
		std::vector< cv::DMatch > good_matches;

		for (int i = 0; i < cv_desc_object.rows; i++)
		{
			if (matches[i].distance <= std::max(2.5 * min_dist, 0.35))
			{
				good_matches.push_back(matches[i]);
			}
		}

		std::cout << "Good matches: " << good_matches.size() << std::endl;

		//-- Draw only "good" matches
		cv::Mat img_matches;
		
		float fx, fy;
		fx = fy = 1.0f;
		if (up_scale) { fx = fy = 2.0f; }
		cv::Mat show_img_object, show_img_scene;
		cv::resize(img_object, show_img_object, cv::Size(img_object.cols * fx, img_object.rows * fy), fx, fy);
		cv::resize(img_scene, show_img_scene, cv::Size(img_scene.cols * fx, img_scene.rows * fy), fx, fy);


		cv::drawMatches(show_img_object, cv_kp_object, show_img_scene, cv_kp_scene, good_matches, img_matches,
			cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

#ifdef  _RANSAC_

		//-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

        for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(cv_kp_object[good_matches[i].queryIdx].pt);
			scene.push_back(cv_kp_scene[good_matches[i].trainIdx].pt);
		}

		cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

		if (!H.empty())
		{
			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<cv::Point2f> obj_corners(4);
			obj_corners[0] = cv::Point(0, 0); obj_corners[1] = cv::Point(show_img_object.cols, 0);
			obj_corners[2] = cv::Point(show_img_object.cols, show_img_object.rows); obj_corners[3] = cv::Point(0, show_img_object.rows);
			std::vector<cv::Point2f> scene_corners(4);

			perspectiveTransform(obj_corners, scene_corners, H);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(img_matches, scene_corners[0] + cv::Point2f(show_img_object.cols, 0), scene_corners[1] + cv::Point2f(show_img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[1] + cv::Point2f(show_img_object.cols, 0), scene_corners[2] + cv::Point2f(show_img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[2] + cv::Point2f(show_img_object.cols, 0), scene_corners[3] + cv::Point2f(show_img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
			line(img_matches, scene_corners[3] + cv::Point2f(show_img_object.cols, 0), scene_corners[0] + cv::Point2f(show_img_object.cols, 0), cv::Scalar(0, 255, 0), 4);

			//-- Show detected matches
			if (img_matches.rows != 0) {
				imshow("Good Matches & Object detection", img_matches);
				cv::waitKey();
			}
		}
		else
		{
			std::cerr << "Couldn't findHomography ... fixme!! " << std::endl;
		}
#else
		//-- Show detected matches
		if (img_matches.rows != 0) {
			cv::imshow("Found Matches", img_matches);
			cv::waitKey();
		}
#endif //  _RANSAC_		
	}

	void Plot_Img_Matches_ocv(
		cv::Mat& img_object, cv::Mat& img_scene,
		const std::vector<Ipoint>& kp_object,
		const std::vector<Ipoint>& kp_scene,
		const bool up_scale=false)
	{
		std::vector < cv::KeyPoint> cv_kp_object, cv_kp_scene;
		cv::Mat cv_desc_object, cv_desc_scene;

		ConvertToCVKeyPoint(kp_object, cv_kp_object, cv_desc_object);
		ConvertToCVKeyPoint(kp_scene, cv_kp_scene, cv_desc_scene);
		
		//-- Step 2: Matching descriptor vectors with a FLANN based matcher
		// Since SURF is a floating-point descriptor NORM_L2 is used
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
		std::vector< std::vector<cv::DMatch> > knn_matches;
		matcher->knnMatch(cv_desc_object, cv_desc_scene, knn_matches, 2);

		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.75f;
		std::vector<cv::DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		std::cout << "Found stable matches:\t" << good_matches.size() << std::endl;
		if (!good_matches.empty())
		{
			//-- Draw matches
			cv::Mat img_matches;

			float fx, fy;
			fx = fy = 1.0f;
			if (up_scale) { fx = fy = 2.0f; }

			cv::Mat show_img_object, show_img_scene;
			cv::resize(img_object, show_img_object, cv::Size(img_object.cols * fx, img_object.rows * fy), fx, fy);
			cv::resize(img_scene, show_img_scene, cv::Size(img_scene.cols * fx, img_scene.rows * fy), fx, fy);

			drawMatches(show_img_object, cv_kp_object, show_img_scene, cv_kp_scene, good_matches, img_matches, cv::Scalar::all(-1),
				cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Show detected matches
			imshow("Good Matches", img_matches);
			cv::waitKey();
		}
	}
}
