#include "inverted_index.h"

void InvertedIndexList::Initialize(int cluster_num, cv::Mat* hist, std::bitset<BCODE_LEN>* hash_codes) {
	std::bitset<BCODE_LEN> maxx(0xffffffffffffffff);
	cv::Mat labels, centers;
	kmeans(*hist, cluster_num, labels,
		cv::TermCriteria(CV_TERMCRIT_ITER, 20, 1.0),
		3, cv::KMEANS_PP_CENTERS, centers);
	num_ = cluster_num;
	index_ = new Inverted_index[cluster_num];
	int* num = new int[cluster_num];
	for (int i = 0; i < cluster_num; i++)
	{
		num[i] = 0;
		index_[i].center = centers.row(i);
	}
	for (int i = 0; i < (*hist).rows; i++)
	{
		int cluster = labels.at<int>(i);
		num[cluster]++;
		index_[cluster].imageID.push_back(i);
		index_[cluster].hash_code.push_back(hash_codes[i]);
		index_[cluster].hash_code_int.push_back((hash_codes[i] & maxx).to_ullong());
	}
	for (int i = 0; i < cluster_num; i++)
	{
		index_[i].num = num[i];
	}
	flann_index_ = new cv::flann::Index(centers, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
}