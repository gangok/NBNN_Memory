#ifndef NBNN_BASIC_H_
#define NBNN_BASIC_H_

#include "feature.h"
#include <string>
#include <vector>

#include <opencv\cv.h>

using namespace std;

class NbnnBasic {
public:
	const static int feature_num_points = 512;
	unsigned __int64 time_center;
	unsigned __int64 time_hashcode;
	void BatchTest();
	
	//accessors and mutators
	string image_folder() const { return image_folder_; }
	void set_image_folder(const string &image_folder) {
		image_folder_ = image_folder;
	}
	int num_of_test_image() const { return num_of_test_image_; }
	void set_num_of_test_image(const int num_of_test_image) {
		num_of_test_image_ = num_of_test_image;
	}
	int num_of_train_image() const { return num_of_train_image_; }
	void set_num_of_train_image(const int num_of_train_image) {
		num_of_train_image_ = num_of_train_image;
	}
	int num_of_class() const { return num_of_class_; }
	void set_num_of_class(const int num_of_class) {
		num_of_class_ = num_of_class;
	}
	int num_of_threads() const { return num_of_threads_; }
	void set_num_of_threads(const int num_of_threads) {
		num_of_threads_ = num_of_threads;
	}
protected:
	void Initialize();
	bool CheckOptions();
	int GetFileList(const char *searchkey, vector<string> &list); // return number of files
	void GetFolderList();
	int GetFeature(string folder, string classname, string filename, Feature* feature); // return number of points
	int GetFeatureOld(string folder, string classname, string filename, Feature* feature,int num_of_points);
	int GetTrainFeature(int class_num, int file_num, Feature* feature) {
		return GetFeatureOld(image_folder_, classes_[class_num], train_file_[class_num][file_num], feature, feature_num_points);
	}
	int GetTestFeature(int class_num, int file_num, Feature* feature) {
		return GetFeatureOld(image_folder_, classes_[class_num], test_file_[class_num][file_num], feature, feature_num_points);
	}
	void PrepareFeatures();
	void MakeFlannIndex(int nn);
	void QueryImagesFlann();
	int ClassifyImageFlann(Feature& feature);
	void PrintSettings();

	string image_folder_;
	int num_of_test_image_;
	int num_of_train_image_;
	int num_of_class_;
	string* classes_;
	string** train_file_;
	string** test_file_;
	Feature** train_features_;
	int total_num_of_train_features_;
	int* num_of_train_features_;
	Feature** test_features_;
	int total_num_of_test_features_;
	int* num_of_test_features_;
	int num_of_threads_;
	cv::flann::Index** flann_index_;
};

#endif