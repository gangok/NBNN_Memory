#ifndef FEATURE_H_
#define FEATURE_H_

#include <opencv\cv.h>
#include <vector>

class Feature {
public:
	~Feature() {
		for(int i=0;i<num_of_points_;i++)
			delete data[i];
		delete data;
	}
	int num_of_points_;
	int dimension_;
	float** data;
};

class Feature_mat {
public:
	~Feature_mat() {
		for(int i=0;i<num_of_points_;i++)
			delete data[i];
		delete data;
		for(int i=0;i<num_of_points_;i++)
			data_mat[i].release();
	}
	int num_of_points_;
	int dimension_;
	float** data;
	cv::Mat* data_mat;
	std::vector< std::vector<float> > data_vec;
};
#endif