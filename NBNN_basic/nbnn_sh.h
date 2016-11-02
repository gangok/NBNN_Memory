#ifndef NBNN_SH_H_
#define NBNN_SH_H_

#include <bitset>
#include "nbnn_basic.h"
#include "Parameters.h"
#include "inverted_index.h"

class NbnnSH : public NbnnBasic {
public:
	void BatchTest();
	//accessors and mutators
	int indexing_cluster_num() const { return indexing_cluster_num_; }
	void set_indexing_cluster_num(const int indexing_cluster_num) {
		indexing_cluster_num_ = indexing_cluster_num;
	}
	double distance_alpha() const { return distance_alpha_; }
	void set_distance_alpha(const double distance_alpha) {
		distance_alpha_ = distance_alpha;
	}
protected:
	void Initialize();
	bool CheckOptions();
	void SetSpheresAndCodeInOneSpace();
	void SetSpheresAndCodeInEachSpace();
	void MakeSHIndex(int cluster_num);
	void QueryImagesSH();
	void PrepareFeatures();
	int ClassifyImageSH(int index_class, int index_file, Feature_mat& feature);
	void PrintSettings();
	double CalculateEuclideanDistance(float* a,float* b,int size) {
		return CalculateEuclideanDistance(a, b, size, 0, 0, 0, 0);
	}
	double CalculateEuclideanDistance(float* a,float* b,int size, double x1, double y1, double x2, double y2);
	double CalculateEuclideanDistance2(float* a,cv::Mat center,int size);

	int GetFeatureOld(string folder, string classname, string filename, Feature_mat* feature,int num_of_points);
	int GetTrainFeature(int class_num, int file_num, Feature_mat* feature) {
		return GetFeatureOld(image_folder_, classes_[class_num], train_file_[class_num][file_num], feature, feature_num_points);
	}
	int GetTestFeature(int class_num, int file_num, Feature_mat* feature) {
		return GetFeatureOld(image_folder_, classes_[class_num], test_file_[class_num][file_num], feature, feature_num_points);
	}

	int get_index_labeled(int nclass,int nfile,int nlast);
	int get_index_test(int nclass,int nfile,int nlast);

	int indexing_cluster_num_;
	double distance_alpha_;

	std::bitset<BCODE_LEN>* hash_codes_labeled_;
	std::bitset<BCODE_LEN>* hash_codes_test_;
	unsigned __int64* hash_codes_int_test_;
	std::bitset<BCODE_LEN>** hash_codes_test_each_space_;

	Feature_mat** train_features_;
	Feature_mat** test_features_;

	InvertedIndexList* bookmarks_;
};

#endif