#include "nbnn_basic.h"
#include "nbnn_basic_local.h"
#include "nbnn_sh.h"
#include "nbnn_sh_local.h"

#include <iostream>

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cout << "you should provide at least 1 argument(basic or basiclocal or sh)\n";
		return 1;
	} else {
		if (strcmp(argv[1], "basic") == 0) {
			NbnnBasic nbnnb;
			nbnnb.set_image_folder("101_ObjectCategories_dsift");
			nbnnb.set_num_of_train_image(15);
			nbnnb.set_num_of_test_image(15);
			nbnnb.set_num_of_class(101);
			if(argc == 4) {// nbnn.exe basic 101_object_dsift_sp6 101
				nbnnb.set_image_folder(argv[2]);
				nbnnb.set_num_of_class(atoi(argv[3]));
			} else if(argc == 7) {// nbnn.exe basic 101_object_dsift_sp6 15 15 101 24
				nbnnb.set_image_folder(argv[2]);
				nbnnb.set_num_of_train_image(atoi(argv[3]));
				nbnnb.set_num_of_test_image(atoi(argv[4]));
				nbnnb.set_num_of_class(atoi(argv[5]));
				nbnnb.set_num_of_threads(atoi(argv[6]));
			} else {
				std::cout << "number of arguments should be 3 or 5 in basic mode\n";
				return 1;
			}
			nbnnb.BatchTest();
		} else if (strcmp(argv[1], "basiclocal") == 0) {
			NbnnBasicLocal nbnnb;
			nbnnb.set_image_folder("101_ObjectCategories_dsift_sp6");
			nbnnb.set_num_of_train_image(15);
			nbnnb.set_num_of_test_image(15);
			nbnnb.set_num_of_class(101);
			if(argc == 4) {// nbnn.exe basic 101_object_dsift_sp6 101
				nbnnb.set_image_folder(argv[2]);
				nbnnb.set_num_of_class(atoi(argv[3]));
			} else if(argc == 7) {// nbnn.exe basiclocal 101_object_dsift_sp6 15 15 101 24
				nbnnb.set_image_folder(argv[2]);
				nbnnb.set_num_of_train_image(atoi(argv[3]));
				nbnnb.set_num_of_test_image(atoi(argv[4]));
				nbnnb.set_num_of_class(atoi(argv[5]));
				nbnnb.set_num_of_threads(atoi(argv[6]));
			} else {
				std::cout << "number of arguments should be 3 or 5 in basic mode\n";
				return 1;
			}
			nbnnb.BatchTest();
		} else if (strcmp(argv[1], "sh") == 0) {
			NbnnSH nbnnsh;
			nbnnsh.set_image_folder("101_ObjectCategories_dsift");
			nbnnsh.set_num_of_test_image(15);
			nbnnsh.set_num_of_train_image(15);
			nbnnsh.set_num_of_class(101);
			nbnnsh.set_indexing_cluster_num(30);
			nbnnsh.set_distance_alpha(0.5);
			if(argc == 4) {// nbnn.exe sh 101_object_dsift_sp6 101
				nbnnsh.set_image_folder(argv[2]);
				nbnnsh.set_num_of_class(atoi(argv[3]));
			} else if(argc == 9) {// nbnn.exe sh 101_object_dsift_sp6 15 15 101 30 0.5 24
				nbnnsh.set_image_folder(argv[2]);
				nbnnsh.set_num_of_train_image(atoi(argv[3]));
				nbnnsh.set_num_of_test_image(atoi(argv[4]));
				nbnnsh.set_num_of_class(atoi(argv[5]));
				nbnnsh.set_indexing_cluster_num(atoi(argv[6]));
				nbnnsh.set_distance_alpha(atof(argv[7]));
				nbnnsh.set_num_of_threads(atof(argv[8]));
			} else {
				std::cout << "number of arguments should be 3 or 7 in sh mode\n";
				return 1;
			}
			nbnnsh.BatchTest();
		} else if (strcmp(argv[1], "shlocal") == 0) {
			NbnnSHLocal nbnnsh;
			nbnnsh.set_image_folder("101_ObjectCategories_dsift");
			nbnnsh.set_num_of_test_image(15);
			nbnnsh.set_num_of_train_image(15);
			nbnnsh.set_num_of_class(101);
			nbnnsh.set_indexing_cluster_num(30);
			if(argc == 4) {// nbnn.exe shlocal 101_object_dsift_sp6 101
				nbnnsh.set_image_folder(argv[2]);
				nbnnsh.set_num_of_class(atoi(argv[3]));
			} else if(argc == 8) {// nbnn.exe shlocal 101_object_dsift_sp6 15 15 101 30 24
				nbnnsh.set_image_folder(argv[2]);
				nbnnsh.set_num_of_train_image(atoi(argv[3]));
				nbnnsh.set_num_of_test_image(atoi(argv[4]));
				nbnnsh.set_num_of_class(atoi(argv[5]));
				nbnnsh.set_indexing_cluster_num(atoi(argv[6]));
				nbnnsh.set_num_of_threads(atoi(argv[7]));
			} else {
				std::cout << "number of arguments should be 3 or 6 in shlocal mode\n";
				return 1;
			}
			nbnnsh.BatchTest();
		} else	{
			std::cout << "you should provide at least 1 argument(basic or sh)\n";
			return 1;
		}
	}
}