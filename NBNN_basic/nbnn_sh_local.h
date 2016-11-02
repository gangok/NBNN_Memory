#ifndef NBNN_SH_LOCAL_H_
#define NBNN_SH_LOCAL_H_

#include "nbnn_sh.h"

class NbnnSHLocal : public NbnnSH {
public:
	void BatchTest();
protected:
	void Initialize();
	void MakeSHIndex(int cluster_num);
	void QueryImagesSH();
	int ClassifyImageSH(int index_class, int index_file, Feature_mat& feature);
	
	InvertedIndexList bookmark_;
};

#endif