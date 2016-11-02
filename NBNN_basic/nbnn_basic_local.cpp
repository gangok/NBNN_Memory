#include "nbnn_basic_local.h"

#include <iostream>
#include <time.h>
#include <omp.h>

#include <windows.h>

using namespace std;

void NbnnBasicLocal::BatchTest() {
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

int NbnnBasicLocal::ClassifyImageFlann(Feature& feature) {
	int nn = 21;
	vector<int> index(128);
	vector<float> dist(128);
	float* sum_dist = new float[num_of_class_];
	for(int i=0;i<num_of_class_;i++)
		sum_dist[i] = 0;
	LARGE_INTEGER start, mid, end;
	for(int x=0;x<feature.num_of_points_;x++) {
		vector<float> input(feature.data[x], feature.data[x] + 128);
		QueryPerformanceCounter(&start);
		flann_index_->knnSearch(input, index, dist, nn, cv::flann::SearchParams(128));
		QueryPerformanceCounter(&mid);
		float distb = dist[nn-1] * dist[nn-1];
		bool* check = new bool[num_of_class_];
		for(int i=0;i<num_of_class_;i++)
			check[i] = false;
		for(int i=0;i<nn-1;i++) {
			int class_num = index[i] / (total_num_of_train_features_ / num_of_class_);
			if(check[class_num] == false) {
				float distc = dist[i] * dist[i];
				sum_dist[class_num] += (distc - distb);
				check[class_num] = true;
			}
		}
		delete check;
		QueryPerformanceCounter(&end);
		time_center += (mid.QuadPart - start.QuadPart);
		time_hashcode += (end.QuadPart - mid.QuadPart);
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

void NbnnBasicLocal::MakeFlannIndex(int nn) { // assumes SIFT
	cv::Mat* dataset = new cv::Mat(total_num_of_train_features_,128, CV_32F);
	int index = 0;
	for(int x=0;x<num_of_class_;x++)
	{
		for(int i=0;i<num_of_train_image_;i++) {
			const Feature* feature= &(train_features_[x][i]);
			for(int j=0;j<feature->num_of_points_;j++) {
				for(int k=0;k<128;k++) {
					(*dataset).at<float>(index,k) = feature->data[j][k];
				}
				index++;
			}
		}
	}
	flann_index_ = new cv::flann::Index(*dataset, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
}

void NbnnBasicLocal::QueryImagesFlann() {
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