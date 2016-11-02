#ifndef NBNN_BASIC_LOCAL_H_
#define NBNN_BASIC_LOCAL_H_

#include "nbnn_basic.h"

class NbnnBasicLocal : public NbnnBasic {
public:
	void BatchTest();
protected:
	void MakeFlannIndex(int nn);
	void QueryImagesFlann();
	int ClassifyImageFlann(Feature& feature);
	
	cv::flann::Index* flann_index_;
};

#endif