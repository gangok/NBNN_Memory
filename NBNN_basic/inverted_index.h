#ifndef INVERTED_INDEX_H_
#define INVERTED_INDEX_H_

#include <opencv2\opencv.hpp>
#include "Parameters.h"
#include <bitset>

struct Inverted_index
{
	int num;
	cv::Mat center;
	std::vector<int> imageID;
	std::vector<std::bitset<BCODE_LEN>> hash_code;
	std::vector<unsigned __int64> hash_code_int;
};

class InvertedIndexList
{
public:
	void Initialize(int cluster_num, cv::Mat* hist, std::bitset<BCODE_LEN>* hash_codes);
	int num_;
	Inverted_index* index_;
	cv::flann::Index* flann_index_;
};

#endif