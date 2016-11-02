#include "nbnn_sh.h"

#include <time.h>
#include <iostream>
#include <fstream>

#include <bitset>
#include "Parameters.h"
#include "BinaryHash.h"
#include <omp.h>

#include <Windows.h>

using namespace std;

void Initialize_Data(REAL_TYPE** arr, REAL_TYPE** q_arr,int nData, int nQData, int dim);
bitset<BCODE_LEN>* get_SH_code_data();
bitset<BCODE_LEN>* get_SH_code_test();
void ReleaseMemory();

void NbnnSH::Initialize() {
	classes_ = new string[num_of_class_];
	train_file_ = new string*[num_of_class_];
	test_file_ = new string*[num_of_class_];
	for(int i=0;i<num_of_class_;i++) {
		train_file_[i] = new string[num_of_train_image_];
		test_file_[i] = new string[num_of_test_image_];
	}
	num_of_train_features_ = new int[num_of_class_];
	num_of_test_features_ = new int[num_of_class_];
	bookmarks_ = new InvertedIndexList[num_of_class_];
	hash_codes_test_each_space_ = new bitset<BCODE_LEN>*[num_of_class_];
}

void NbnnSH::BatchTest() {
	LARGE_INTEGER l_start, l_read_features, l_make_index, l_end;
	if (CheckOptions()== false)
		return;
	cout << "batch test start!\n";
	time_t t_start = time(NULL);
	QueryPerformanceCounter(&l_start);

	string outputfile = "logs\\output_";
	char temp[20];
	outputfile += itoa(t_start,temp,10);
	outputfile += ".txt";
	freopen(outputfile.c_str(),"w", stdout);

	cout << typeid(this).name() << image_folder_ << endl;
	Initialize();
	GetFolderList();
	PrepareFeatures();
	time_t t_read_features = time(NULL);
	QueryPerformanceCounter(&l_read_features);
	cout << "total number of features : " << total_num_of_train_features_ << endl;
	cout << "time for reading features : " << t_read_features - t_start << endl;
	cout << "L for reading features : " << l_read_features.QuadPart - l_start.QuadPart << endl;
	SetSpheresAndCodeInOneSpace();
	MakeSHIndex(indexing_cluster_num_);
	time_t t_make_index = time(NULL);
	QueryPerformanceCounter(&l_make_index);
	cout << "making index finished\n";
	cout << "time for indexing : " << t_make_index - t_read_features << endl;
	cout << "L for indexing : " << l_make_index.QuadPart - l_read_features.QuadPart << endl;
	time_center = 0;
	time_hashcode = 0;
	QueryImagesSH();
	time_t t_end = time(NULL);
	QueryPerformanceCounter(&l_end);
	cout << "time for querying : " << t_end - t_make_index << endl;
	cout << "L for querying : " << l_end.QuadPart - l_make_index.QuadPart << endl;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	cout << "time for center : " << time_center << endl;
	cout << "time for hashcode : " << time_hashcode << endl;
	cout << "time frequency : " << frequency.QuadPart << endl;
	PrintSettings();
	freopen("CON","w", stdout);
}

int NbnnSH::GetFeatureOld(string folder, string classname, string filename, Feature_mat* feature, int num_of_points) {
	feature->dimension_ = 128;
	feature->num_of_points_ = num_of_points;
	feature->data_mat = new cv::Mat[num_of_points];
	feature->data = new float*[num_of_points];
	float** data = feature->data;
	std::vector<float>* temp_vec;
	for(int i=0;i<num_of_points;i++) {
		data[i] = new float[feature->dimension_];
		feature->data_mat[i].create(1, feature->dimension_, CV_32F);
		temp_vec = new std::vector<float>(128);
		feature->data_vec.push_back(*temp_vec);
	}
	//if file exists, read sift from file
	string path = folder + "\\" + classname + "\\" + filename;
	ifstream fin(path, ios::binary);
	if(fin.is_open()) {
		for(int x=0;x<num_of_points;x++) {
			for(int y=0;y<128;y++) {
				fin.read((char*)&(data[x][y]), sizeof(float));
				feature->data_mat[x].at<float>(0,y) = data[x][y];
				feature->data_vec[x][y] = data[x][y];
			}
		}
		fin.close();
		return feature->num_of_points_;
	} else {
		cout << "There is no such file\n";
		return NULL;
	}
}

void NbnnSH::PrepareFeatures() {
	train_features_ = new Feature_mat*[num_of_class_];
	total_num_of_train_features_ = 0;
	for(int i=0;i<num_of_class_;i++) {
		train_features_[i] = new Feature_mat[num_of_train_image_];
		num_of_train_features_[i] = 0;
	}
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_train_image_;j++) {
			num_of_train_features_[i] += GetTrainFeature(i,j,&(train_features_[i][j]));
		}
		total_num_of_train_features_ += num_of_train_features_[i];
	}

	test_features_ = new Feature_mat*[num_of_class_];
	total_num_of_test_features_ = 0;
	for(int i=0;i<num_of_class_;i++) {
		test_features_[i] = new Feature_mat[num_of_test_image_];
		num_of_test_features_[i] = 0;
	}
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_test_image_;j++) {
			num_of_test_features_[i] += GetTestFeature(i,j,&(test_features_[i][j]));
		}
		total_num_of_test_features_ += num_of_test_features_[i];
	}
}

void NbnnSH::SetSpheresAndCodeInOneSpace() {
	float** labeled_data;
	float** test_data;
	labeled_data = new float*[total_num_of_train_features_];
	int count = 0;
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_train_image_;j++) {
			for(int k=0;k<train_features_[i][j].num_of_points_;k++)
			{
				labeled_data[count] = new float[128];
				memcpy(labeled_data[count],train_features_[i][j].data[k],sizeof(float) * 128);
				count++;
			}
		}
	}
	test_data = new float*[total_num_of_test_features_];
	count = 0;
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_test_image_;j++) {
			for(int k=0;k<test_features_[i][j].num_of_points_;k++)
			{
				test_data[count] = new float[128];
				memcpy(test_data[count],test_features_[i][j].data[k],sizeof(float) * 128);
				count++;
			}
		}
	}
	Initialize_Data(labeled_data, test_data, total_num_of_train_features_ , total_num_of_test_features_, 128);
	hash_codes_labeled_ = get_SH_code_data();
	hash_codes_test_ = get_SH_code_test();
	hash_codes_int_test_= new unsigned __int64[total_num_of_test_features_];
	for(int i=0;i<total_num_of_test_features_;i++) {
		hash_codes_int_test_[i] = (hash_codes_test_[i] & maxx).to_ullong();
	}
	ReleaseMemory();
}

int NbnnSH::get_index_labeled(int nclass,int nfile,int nlast) {
	int index_file = nclass * num_of_train_image_ + nfile;
	int index = index_file * feature_num_points + nlast;
	return index;
}

int NbnnSH::get_index_test(int nclass,int nfile,int nlast) {
	int index_file = nclass * num_of_test_image_ + nfile;
	int index = index_file * feature_num_points + nlast;
	return index;
}

void NbnnSH::MakeSHIndex(int cluster_num) {
	for(int x=0;x<num_of_class_;x++)
	{
		cv::Mat* dataset = new cv::Mat(num_of_train_features_[x],128, CV_32F);
		int count = 0;
		for(int i=0;i<num_of_train_image_;i++) {
			for(int j=0;j<train_features_[x][i].num_of_points_;j++) {
				for(int k=0;k<128;k++) {
					(*dataset).at<float>(count,k) = train_features_[x][i].data[j][k];
				}
				count++;
			}
		}
		bookmarks_[x].Initialize(cluster_num, dataset, &hash_codes_labeled_[get_index_labeled(x,0,0)]);
	}
}

double NbnnSH::CalculateEuclideanDistance(float* a,float* b,int size, double x1, double y1, double x2, double y2) {
	double ret = 0;
	for(int i=0;i<size;i++)
		ret = ret + (b[i] - a[i]) * (b[i] - a[i]);
	ret += distance_alpha_ * distance_alpha_ * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	ret = sqrt(ret);
	return ret;
}

double NbnnSH::CalculateEuclideanDistance2(float* a,cv::Mat center,int size) {
	double ret = 0;
	for(int i=0;i<size;i++)
		ret = ret + (center.at<float>(0, i) - a[i]) * (center.at<float>(0, i) - a[i]);
	ret = sqrt(ret);
	return ret;
}

int NbnnSH::ClassifyImageSH(int index_class, int index_file, Feature_mat& feature) {
	int result;
	float* sum_dist = new float[num_of_class_];
	double min_Hdistance, distance;
	double min_center_distance;
	for(int i=0;i<num_of_class_;i++)
		sum_dist[i] = 0;
	LARGE_INTEGER start, mid, end;
	for(int x=0;x<feature.num_of_points_;x++) {
		for(int i=0;i<num_of_class_;i++) {
			QueryPerformanceCounter(&start);
			int min_center = 0;
			min_center_distance = CalculateEuclideanDistance2(feature.data[x],bookmarks_[i].index_[0].center,128);
			for(int j=1;j<bookmarks_[i].num_;j++) {
				distance = CalculateEuclideanDistance2(feature.data[x],bookmarks_[i].index_[j].center,128);
				if(min_center_distance > distance) {
					min_center = j;
					min_center_distance = distance;
				}
			}
			min_Hdistance = 0;
			QueryPerformanceCounter(&mid);
			min_Hdistance = Compute_HD(hash_codes_int_test_[get_index_test(index_class,index_file,x)]
												,bookmarks_[i].index_[min_center].hash_code_int[0]);
			int min_j=0;
			for(int j=0;j<bookmarks_[i].index_[min_center].num;j++) {
				distance = Compute_HD(hash_codes_int_test_[get_index_test(index_class,index_file,x)]
											,bookmarks_[i].index_[min_center].hash_code_int[j]);
				if (distance < min_Hdistance) {
					min_Hdistance = distance;
					min_j = j;
				}
			}
			sum_dist[i] += min_Hdistance;
			QueryPerformanceCounter(&end);
			time_center += (mid.QuadPart - start.QuadPart);
			time_hashcode += (end.QuadPart - mid.QuadPart);
		}
	}
	
	int ret = 0;
	double min_dist = sum_dist[0];
	for(int i=0;i<num_of_class_;i++) {
		if (sum_dist[i] < min_dist) {
			min_dist = sum_dist[i];
			ret = i;
		}
	}
	return ret;
}

void NbnnSH::QueryImagesSH() {
	int** classified;
	classified = new int*[num_of_class_];
	for(int i=0;i<num_of_class_;i++) {
		classified[i] = new int[num_of_test_image_];
	}
	
	if(num_of_threads() != 0)
		omp_set_num_threads(num_of_threads());
	#pragma omp parallel
	{
		if(omp_get_thread_num() == 0)
			cout << "num threads : " << omp_get_num_threads() << endl;
		#pragma omp for
		for(int i=0;i<num_of_class_;i++) {
			int th_id = omp_get_thread_num();
			int count = 0;
			for(int j=0;j<num_of_test_image_;j++) {
				Feature_mat* feature = &test_features_[i][j];
				classified[i][j] = ClassifyImageSH(i,j, *feature);
				count++;
				time_t now = time(NULL);
				cout << ctime(&now) << "[" << th_id << "," << count << "]" << "classified into " << classified[i][j] << "(" << i << ")" << endl;
			}
		}
	}

	int right, wrong;
	int sum_right = 0,sum_wrong = 0;
	for(int i=0;i<num_of_class_;i++) {
		right = 0;
		wrong = 0;
		cout << classes_[i] << ":" << endl;
		for(int j=0;j<num_of_test_image_;j++) {
			if(classified[i][j] == i)
				right++;
			else
				wrong++;
		}
		cout << "accuracy:" << (double)right / (right+wrong) * 100 << "%(" << right << "/" << num_of_test_image_ << ")" << endl;
		sum_right += right;
		sum_wrong += wrong;
	}
	cout << "total:" << (double)sum_right / (sum_right+sum_wrong) * 100 << "%" << endl;
	cout << "right:" << sum_right << "wrong:" << sum_wrong << endl;
}

void NbnnSH::PrintSettings() {
	NbnnBasic::PrintSettings();
	cout << "setting : \n";
	cout << "bit code length : " << BCODE_LEN << endl;
	cout << "number of clusters in indexing : " << indexing_cluster_num() << endl;
	cout << "distance alpha value : " << distance_alpha() << endl;
}

bool NbnnSH::CheckOptions() {
	if (indexing_cluster_num_ <= 0) {
		cout << "Number of clusters in indexing should be bigger than 0!\n";
		return false;
	}
	return NbnnBasic::CheckOptions();
}