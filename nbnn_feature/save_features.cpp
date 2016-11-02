#include <vl/generic.h>
#include <vl/dsift.h>
#include <windows.h>
#include <stdlib.h>
#include <time.h>

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\nonfree\features2d.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <omp.h>

using namespace std;
using namespace cv;

#define SAVE_FEATURES_gridNumX 16
#define SAVE_FEATURES_gridNumY 16

//#define SAVE_FEATURES_N_CLASS 100
//const int n_class  = 100;

ofstream out;

float** getDenseSift2(String folder, String filename, int bin_size) {
	Mat image = imread(folder + filename, CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<float> imgvec;
	for (int i = 0; i < image.rows; ++i){
		for (int j = 0; j < image.cols; ++j){
			imgvec.push_back(image.at<unsigned char>(i,j) / 255.0f);																															
		}
	}
	VlDsiftFilter* dsfilter = vl_dsift_new_basic(image.cols, image.rows, 5, bin_size);
	int minX = 0;
	int minY = 0;
	int maxX = image.cols - 1;
	int maxY = image.rows - 1;
	int stepX = (image.cols-bin_size*3) / SAVE_FEATURES_gridNumX;
	if ((image.cols-bin_size*3) % SAVE_FEATURES_gridNumX > 0) {
		stepX++;
		int gridnum = (image.cols-bin_size*3) / stepX;
		if ((image.cols-bin_size*3) % stepX > 0)
			gridnum++;
		if(gridnum < SAVE_FEATURES_gridNumX) {
			stepX--;
			int lastpoint = bin_size*1.5 + 15 * stepX;
			lastpoint += bin_size*1.5 + 1;
			maxX = lastpoint;
		}
	}
	int stepY = (image.rows-bin_size*3) / SAVE_FEATURES_gridNumY;
	if ((image.rows-bin_size*3) % SAVE_FEATURES_gridNumY > 0) {
		stepY++;
		int gridnum = (image.rows-bin_size*3) / stepY;
		if ((image.rows-bin_size*3) % stepY > 0)
			gridnum++;
		if(gridnum < SAVE_FEATURES_gridNumY) {
			stepY--;
			int lastpoint = bin_size*1.5 + 15 * stepY;
			lastpoint += bin_size*1.5 + 1;
			maxY = lastpoint;
		}
	}
	if (stepX <= 0 || stepY <= 0) return NULL;
	vl_dsift_set_bounds(dsfilter,minX,minY,maxX,maxY);
	vl_dsift_set_steps(dsfilter, stepX, stepY);	
	// call processing function of vl
	vl_dsift_process(dsfilter, &imgvec[0]);
	int keypointnum = vl_dsift_get_keypoint_num(dsfilter);
	int descriptornum = vl_dsift_get_descriptor_size(dsfilter);
	const float* features = vl_dsift_get_descriptors(dsfilter);
	// echo number of keypoints found
	cout << folder << filename << ':' << image.cols << ',' << image.rows << ',' << keypointnum << "," << descriptornum << std::endl;
	/*const VlDsiftKeypoint* keypoints = vl_dsift_get_keypoints(dsfilter);
	for(int i=0;i<keypointnum;i++)
		cout << keypoints[i].x << ' ' << keypoints[i].y << endl;*/
	if (keypointnum < SAVE_FEATURES_gridNumX * SAVE_FEATURES_gridNumY) {
		cout << "Error!" << folder << filename << endl;
		out << "Error!" << folder << filename << endl;
		vl_dsift_delete(dsfilter);
		return NULL;
	}
	float** ret = new float*[SAVE_FEATURES_gridNumX * SAVE_FEATURES_gridNumY];
	for(int i=0;i<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;i++) {
		ret[i] = new float[128];
		memcpy(ret[i],features+i*128,sizeof(float) * 128);
	}
	
	vl_dsift_delete(dsfilter);
	return ret;
}

void free_feature(float** feature) {
	for(int i=0;i<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;i++) {
		delete feature[i];
	}
	delete feature;
}

int GetDirectoryList(const char *searchkey, std::vector<std::string> &list)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(searchkey,&fd);
 
    if(h == INVALID_HANDLE_VALUE)
    {
        return 0; // no files found
    }
 
    while(1)
    {
		if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			list.push_back(fd.cFileName);
 
        if(FindNextFile(h, &fd) == FALSE)
            break;
    }
    return list.size();
}

int GetFileList(const char *searchkey, std::vector<std::string> &list)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(searchkey,&fd);
 
    if(h == INVALID_HANDLE_VALUE)
    {
        return 0; // no files found
    }
 
    while(1)
    {
		if(!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			int len = strlen(fd.cFileName);
			if(len > 3 && (strcmp(fd.cFileName + len - 4, ".jpg") || strcmp(fd.cFileName + len - 4, ".png")))
				list.push_back(fd.cFileName);
		}
        if(FindNextFile(h, &fd) == FALSE)
            break;
    }
    return list.size();
}

void SaveFeaturesToFile(String imageFolder, String dataFolder) {
	//set class name
	vector<string> list;
	String topfolder = imageFolder;
	topfolder.append("*");
	int n_class = GetDirectoryList(topfolder.c_str(), list) - 2;
	if(n_class > 10000) {
		cout << "too many # of class > 10000";
		return;
	}
	String* classes = new String[n_class];
	for(int i=2;i<n_class+2;i++)
		classes[i-2] = list[i];

	CreateDirectory(dataFolder.c_str(), NULL);
	//set file lists
	
	#pragma omp parallel
	{
		if(omp_get_thread_num() == 0)
			cout << "num threads : " << omp_get_num_threads() << endl;
		#pragma omp for
		for(int i=0;i<n_class;i++) {
			vector<string> list;
			String subfolder = imageFolder;
			subfolder.append(classes[i]).append("\\*");
			GetFileList(subfolder.c_str(), list);
			int count = 0;
			int j=0;
			for(j=0;j<list.size();j++) {
				String dirname = dataFolder + classes[i];
				String filename = dirname + "\\" + list[j];
				CreateDirectory(dirname.c_str(), NULL);
				ofstream fout(filename.append(".dat"), ios::binary);
				float** features = getDenseSift2(imageFolder + classes[i] + "\\", list[j], 4);
				if(features == NULL) {
					fout.close();
					count++;
					continue;
				}
				for(int x=0;x<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;x++) {
					for(int y=0;y<128;y++) {
						fout.write((char*)&(features[x][y]), sizeof(float));
					}
				}
				free_feature(features);
				features = getDenseSift2(imageFolder + classes[i] + "\\", list[j], 6);
				if(features == NULL) {
					fout.close();
					count++;
					continue;
				}
				for(int x=0;x<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;x++) {
					for(int y=0;y<128;y++) {
						fout.write((char*)&(features[x][y]), sizeof(float));
					}
				}
				free_feature(features);
				/*features = getDenseSift2(caltechFolder + classes[i] + "\\", list[j], 9);
				for(int x=0;x<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;x++) {
					for(int y=0;y<128;y++) {
						fout.write((char*)&(features[x][y]), sizeof(float));
					}
				}
				features = getDenseSift2(caltechFolder + classes[i] + "\\", list[j
				free_feature(features);], 14);
				for(int x=0;x<SAVE_FEATURES_gridNumX*SAVE_FEATURES_gridNumY;x++) {
					for(int y=0;y<128;y++) {
						fout.write((char*)&(features[x][y]), sizeof(float));
					}
				}
				free_feature(features);*/
				fout.close();
			}
			out << "class[" << i << "]:" << classes[i] << " completed with " << j << "images with" << count << "exceptions" << endl;
		}
	}
}

int main() {
	out.open("log.txt");
	time_t start = time(NULL);
	out << "start time : " << start << endl;
	SaveFeaturesToFile("train\\", "train_DSIFT2\\");
	cout << "Saving features of all images is finished!" << endl;
	time_t end = time(NULL);
	out << "end time : " << end << endl;
	cout << "time_feature : " << end - start << endl;
	out << "time_feature : " << end - start << endl;
	out.close();
}