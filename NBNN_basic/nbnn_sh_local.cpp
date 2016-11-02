#include "nbnn_sh_local.h"

#include <bitset>
#include "Parameters.h"
#include "BinaryHash.h"
#include <omp.h>
#include <queue>

#include <Windows.h>

void NbnnSHLocal::Initialize() {
	classes_ = new string[num_of_class_];
	train_file_ = new string*[num_of_class_];
	test_file_ = new string*[num_of_class_];
	for(int i=0;i<num_of_class_;i++) {
		train_file_[i] = new string[num_of_train_image_];
		test_file_[i] = new string[num_of_test_image_];
	}
	num_of_train_features_ = new int[num_of_class_];
	num_of_test_features_ = new int[num_of_class_];
}

void NbnnSHLocal::MakeSHIndex(int cluster_num) {
	cv::Mat* dataset = new cv::Mat(total_num_of_train_features_,128, CV_32F);
	int count = 0;
	for(int x=0;x<num_of_class_;x++)
	{
		for(int i=0;i<num_of_train_image_;i++) {
			for(int j=0;j<train_features_[x][i].num_of_points_;j++) {
				for(int k=0;k<128;k++) {
					(*dataset).at<float>(count,k) = train_features_[x][i].data[j][k];
				}
				count++;
			}
		}
	}
	bookmark_.Initialize(cluster_num, dataset, &hash_codes_labeled_[get_index_labeled(0,0,0)]);
}

int NbnnSHLocal::ClassifyImageSH(int index_class, int index_file, Feature_mat& feature) {
	int nn = 11;
	//int num_of_centers = 1; need to be implemented
	vector<int> index(128);
	vector<float> dist(128);
	int result;
	float* sum_dist = new float[num_of_class_];
	double distance;
	double min_center_distance;
	for(int i=0;i<num_of_class_;i++)
		sum_dist[i] = 0;
	LARGE_INTEGER start, mid, end;
	for(int x=0;x<feature.num_of_points_;x++) {
		multimap<float,int> nn_map;
		QueryPerformanceCounter(&start);
		bookmark_.flann_index_->knnSearch(feature.data_vec[x], index, dist, 1, cv::flann::SearchParams(128));
		int min_center = index[0];
		QueryPerformanceCounter(&mid);
		for(int j=0;j<nn;j++) {
			distance = Compute_HD(hash_codes_int_test_[get_index_test(index_class,index_file,x)]
											,bookmark_.index_[min_center].hash_code_int[j]);
			nn_map.insert(make_pair(distance,j));
		}
		for(int j=nn;j<bookmark_.index_[min_center].num;j++) {
			distance = Compute_HD(hash_codes_int_test_[get_index_test(index_class,index_file,x)]
										,bookmark_.index_[min_center].hash_code_int[j]);
			nn_map.insert(make_pair(distance,j));
			nn_map.erase(--nn_map.end());
		}
		float distb;
		distb = (--nn_map.end())->first;
		distb = distb * distb;
		bool* check = new bool[num_of_class_];
		for(int i=0;i<num_of_class_;i++)
			check[i] = false;
		std::multimap<float,int>::iterator it=nn_map.begin(); 
		for(int i=0;i<nn-1;i++) {
			float distc;
			int imageID = bookmark_.index_[min_center].imageID[it->second];
			int class_num = imageID / (total_num_of_train_features_ / num_of_class_);
			distc = it->first;
			
			if(check[class_num] == false) {
				distc = distc * distc;
				sum_dist[class_num] += (distc - distb);
				check[class_num] = true;
			}
			it++;
		}
		delete check;
		QueryPerformanceCounter(&end);
		time_center += (mid.QuadPart - start.QuadPart);
		time_hashcode += (end.QuadPart - mid.QuadPart);
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

void NbnnSHLocal::QueryImagesSH() {
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

void NbnnSHLocal::BatchTest() {
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