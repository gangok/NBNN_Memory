#include "nbnn_basic.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <windows.h>
#include <time.h>
#include <omp.h>

#include <opencv\cv.h>

using namespace std;

void NbnnBasic::BatchTest() {
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
	MakeFlannIndex(1);
	time_t t_make_index = time(NULL);
	QueryPerformanceCounter(&l_make_index);
	cout << "making index finished\n";
	cout << "time for indexing : " << t_make_index - t_read_features << endl;
	cout << "L for indexing : " << l_make_index.QuadPart - l_read_features.QuadPart << endl;
	time_center = 0;
	time_hashcode = 0;
	QueryImagesFlann();
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

void NbnnBasic::Initialize() {
	classes_ = new string[num_of_class_];
	train_file_ = new string*[num_of_class_];
	test_file_ = new string*[num_of_class_];
	for(int i=0;i<num_of_class_;i++) {
		train_file_[i] = new string[num_of_train_image_];
		test_file_[i] = new string[num_of_test_image_];
	}
	num_of_train_features_ = new int[num_of_class_];
	num_of_test_features_ = new int[num_of_class_];
	flann_index_ = new cv::flann::Index*[num_of_class_];
}

bool NbnnBasic::CheckOptions() {
	if (image_folder_.empty()) {
		cout << "Image folder name should be set!\n";
		return false;
	}
	if (num_of_test_image_ <= 0 || num_of_train_image_ <= 0) {
		cout << "Number of test/train images should be bigger than 0!\n";
		return false;
	}
	if (num_of_class_ <= 0) {
		cout << "Number of classes image should be bigger than 0!\n";
		return false;
	}
	return true;
}

int NbnnBasic::GetFileList(const char *searchkey, vector<string> &list)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(searchkey,&fd);
 
    if(h == INVALID_HANDLE_VALUE)
    {
        return 0; // no files found
    }
 
    while(1)
    {
        list.push_back(fd.cFileName);
 
        if(FindNextFile(h, &fd) == FALSE)
            break;
    }
    return list.size();
}

void NbnnBasic::GetFolderList() {
	vector<string> list;
	string topfolder = image_folder_;
	topfolder.append("\\*");
	GetFileList(topfolder.c_str(), list);
	for(int i=2;i<num_of_class_+2;i++)
		classes_[i-2] = list[i];
	//set file lists
	int randomSeed = 42;
	srand(randomSeed);
	cout << "random seed : " << randomSeed << endl;
	for(int i=0;i<num_of_class_;i++) {
		list.clear();
		string subfolder = image_folder_;
		subfolder.append("\\").append(classes_[i]).append("\\*");
		GetFileList(subfolder.c_str(), list);
		list.erase(list.begin());
		list.erase(list.begin());
		if(list.size() < num_of_train_image_ + num_of_test_image_) {
			cout << "Error in " << classes_[i] << endl;
			exit(0);
		}
		for(int j=0;j<num_of_train_image_ + num_of_test_image_;j++) {
			int swapindex = rand() % (list.size()-j) + j;
			string temp = list[j];
			list[j] = list[swapindex];
			list[swapindex] = temp;
		}

		for(int j=0;j<num_of_train_image_;j++) {
			train_file_[i][j] = list[j];
		}
		for(int j=0;j<num_of_test_image_;j++) {
			test_file_[i][j] = list[num_of_train_image_ + j];
		}
	}
}

int NbnnBasic::GetFeatureOld(string folder, string classname, string filename, Feature* feature, int num_of_points) {
	feature->dimension_ = 128;
	feature->num_of_points_ = num_of_points;
	feature->data = new float*[num_of_points];
	float** data = feature->data;
	for(int i=0;i<num_of_points;i++) {
		data[i] = new float[feature->dimension_];
	}
	//if file exists, read sift from file
	string path = folder + "\\" + classname + "\\" + filename;
	ifstream fin(path, ios::binary);
	if(fin.is_open()) {
		for(int x=0;x<num_of_points;x++) {
			for(int y=0;y<128;y++) {
				fin.read((char*)&(data[x][y]), sizeof(float));
			}
		}
		fin.close();
		return feature->num_of_points_;
	} else {
		cout << "There is no such file\n";
		return NULL;
	}
}

int NbnnBasic::GetFeature(string folder, string classname, string filename, Feature* feature) {
	string path = folder + "\\" + classname + "\\" + filename;
	
	ifstream fin(path);
	if(fin.is_open()) {
		string type;
		int dimension;
		int num_of_points;
		fin >> type;
		if(type.compare("KOEN1") != 0) {
			cout << "Invalid file type\n";
			return NULL;
		}
		fin >> dimension >> num_of_points;
		feature->dimension_ = dimension;
		feature->num_of_points_ = num_of_points;
		feature->data = new float*[num_of_points];
		float** data = feature->data;
		for(int i=0;i<num_of_points;i++) {
			data[i] = new float[dimension];
		}
		for(int i=0;i<num_of_points;i++) {
			fin >> type;
			fin >> type;
			fin >> type;
			fin >> type;
			fin >> type;
			fin >> type;
			for(int j=0;j<dimension;j++) {
				fin >> type;
				data[i][j] = stoi(type);
			}
		}
		fin.close();
		return num_of_points;
	} else {
		cout << "There is no such file\n";
		return NULL;
	}
}

void NbnnBasic::PrepareFeatures() {
	train_features_ = new Feature*[num_of_class_];
	total_num_of_train_features_ = 0;
	for(int i=0;i<num_of_class_;i++) {
		train_features_[i] = new Feature[num_of_train_image_];
		num_of_train_features_[i] = 0;
	}
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_train_image_;j++) {
			num_of_train_features_[i] += GetTrainFeature(i,j,&(train_features_[i][j]));
		}
		total_num_of_train_features_ += num_of_train_features_[i];
	}

	test_features_ = new Feature*[num_of_class_];
	total_num_of_test_features_ = 0;
	for(int i=0;i<num_of_class_;i++) {
		test_features_[i] = new Feature[num_of_test_image_];
		num_of_test_features_[i] = 0;
	}
	for(int i=0;i<num_of_class_;i++) {
		for(int j=0;j<num_of_test_image_;j++) {
			num_of_test_features_[i] += GetTestFeature(i,j,&(test_features_[i][j]));
		}
		total_num_of_test_features_ += num_of_test_features_[i];
	}
}

void NbnnBasic::MakeFlannIndex(int nn) { // assumes SIFT
	for(int x=0;x<num_of_class_;x++)
	{
		int index = 0;
		cv::Mat* dataset = new cv::Mat(num_of_train_features_[x],128, CV_32F);
		for(int i=0;i<num_of_train_image_;i++) {
			const Feature* feature= &(train_features_[x][i]);
			for(int j=0;j<feature->num_of_points_;j++) {
				for(int k=0;k<128;k++) {
					(*dataset).at<float>(index,k) = feature->data[j][k];
				}
				index++;
			}
		}
		flann_index_[x] = new cv::flann::Index(*dataset, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	}
}

int NbnnBasic::ClassifyImageFlann(Feature& feature) {
	int nn = 1;
	vector<int> index(128);
	vector<float> dist(128);
	float* sum_dist = new float[num_of_class_];
	for(int i=0;i<num_of_class_;i++)
		sum_dist[i] = 0;
	LARGE_INTEGER start, mid, end;
	for(int x=0;x<feature.num_of_points_;x++) {
		for(int i=0;i<num_of_class_;i++) {
			vector<float> input(feature.data[x], feature.data[x] + 128);
			QueryPerformanceCounter(&start);
			flann_index_[i]->knnSearch(input, index, dist, nn, cv::flann::SearchParams(128));
			QueryPerformanceCounter(&mid);
			sum_dist[i] += dist[0] * dist[0];
			QueryPerformanceCounter(&end);
			time_center += (mid.QuadPart - start.QuadPart);
			time_hashcode += (end.QuadPart - mid.QuadPart);
		}
	}
	int ret = 0;
	float min_dist = sum_dist[0];
	for(int i=0;i<num_of_class_;i++) {
		if (sum_dist[i] < min_dist) {
			min_dist = sum_dist[i];
			ret = i;
		}
	}
	return ret;
}

void NbnnBasic::QueryImagesFlann() {
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
				Feature* feature = &test_features_[i][j];
				classified[i][j] = ClassifyImageFlann(*feature);
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
	for(int i=0;i<num_of_class_;i++) {
		delete classified[i];
	}
	delete classified;
}

void NbnnBasic::PrintSettings() {
	cout << "setting : \n";
	cout << "image folder name : " << image_folder_ << endl;
	cout << "number of training images per class : " << num_of_train_image_ << endl;
	cout << "number of test images per class : " << num_of_test_image_ << endl;
	cout << "number of classes : " << num_of_class_ << endl;
}