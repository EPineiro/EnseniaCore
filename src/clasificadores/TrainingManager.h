/*
 * TrainingManager.h
 *
 *  Created on: Jul 7, 2013
 *      Author: EPineiro
 */

#ifndef TRAININGMANAGER_H_
#define TRAININGMANAGER_H_

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "Classifier.h"

#include "../features/SignFeaturesDetector.h"

using namespace std;
using namespace cv;



class TrainingManager {
private:

	int samplesCount;
	int featuresCount;
	int featuresType;

	void scanDirectoryForTrainingImages(string dir, Mat &trainingMat, Mat &labels) {

		DIR *dp;
		struct dirent *dirp;

		int index = 0;

		//primero cargamos las positivas
		std::stringstream positiveDir;
		positiveDir << dir << "/positivas";

		if ((dp = opendir(positiveDir.str().c_str())) == NULL) {
			cout << "Error opening " << dir << endl;
		}

		while ((dirp = readdir(dp)) != NULL) {

			string fileName = string(dirp->d_name);

			if (fileName.find_first_of('.') > 0) {

				std::stringstream imageDir;
				imageDir << dir << "/positivas/" << fileName;

				Mat image = imread(imageDir.str());
				addImageToTrainingMat(trainingMat, index, image, fileName);
				labels.at<float> (index, 0) = 1;
				index++;
			}
		}

		//ahora las negativas
		std::stringstream negativeDir;
		negativeDir << dir << "/negativas";

		if ((dp = opendir(negativeDir.str().c_str())) == NULL) {
			cout << "Error opening " << dir << endl;
		}

		while ((dirp = readdir(dp)) != NULL) {

			string fileName = string(dirp->d_name);

			if (fileName.find_first_of('.') > 0) {

				std::stringstream imageDir;
				imageDir << dir << "/negativas/" << fileName;

				Mat image = imread(imageDir.str());
				addImageToTrainingMat(trainingMat, index, image, fileName);
				labels.at<float> (index, 0) = -1;
				index++;
			}
		}

		closedir(dp);
	}

	void addImageToTrainingMat(Mat &trainingMat, int index, Mat image, string name) {

		SignFeaturesDetector detector;
		vector<float> features = detector.getSignFeatures(image, featuresType, name);

		int i = 0;
		for(vector<float>::const_iterator it = features.begin(); it != features.end(); it++) {

			trainingMat.at<float> (index, i++) = *it;
		}
	}

public:

	inline void setSamplesCount(int count) {
		this->samplesCount = count;
	}

	inline void setFeaturesCount(int count) {
		this->featuresCount = count;
	}

	inline void setFeaturesType(int type) {
		this->featuresType = type;
	}

	void trainClassifier(string imagesDir, Classifier *classifier, string savedClassifierFile) {

		Mat trainingMat(samplesCount, featuresCount, CV_32FC1);
		Mat labelsMat(samplesCount, 1, CV_32FC1);

		scanDirectoryForTrainingImages(imagesDir, trainingMat, labelsMat);

		classifier->train(trainingMat, labelsMat, savedClassifierFile);
	}

	void testClassifier(const string classifierFileName, const vector<string> &samples, Classifier *classifier) {

		classifier->loadClassifier(classifierFileName);

		SignFeaturesDetector detector;

		for(vector<string>::const_iterator it = samples.begin(); it != samples.end(); it++) {

			Mat image = imread(*it);

			string imageName = (*it).substr((*it).find_last_of("/") + 1);

			vector<float> features = detector.getSignFeatures(image, featuresType ,imageName);

			float response = classifier->predict(features, imageName);

			if (fabs(response - 1.0) <= FLT_EPSILON)
			//if (response == 1)
				cout <<"imagen: " <<imageName<< " es la seña" << endl;
			else
				cout <<"imagen: " <<imageName<< " no es la seña" << endl;

		}
	}

	void testClassifier(const string classifierFileName, const string samplesDir, Classifier *classifier) {

		classifier->loadClassifier(classifierFileName);

		SignFeaturesDetector detector;

		DIR *dp;
		struct dirent *dirp;

		if ((dp = opendir(samplesDir.c_str())) == NULL) {
			cout << "Error abriendo directorio con imagenes" << endl;
		}

		while ((dirp = readdir(dp)) != NULL) {

			string fileName = string(dirp->d_name);

			if (fileName.find(".jpg") != string::npos) {

				std::stringstream imageDir;
				imageDir << samplesDir << "/" << fileName;

				Mat image = imread(imageDir.str());

				vector<float> features = detector.getSignFeatures(image, featuresType, fileName);

				float response = classifier->predict(features, fileName);

				if (fabs(response - 1.0) <= FLT_EPSILON)
					//if (response == 1)
					cout << "imagen: " << fileName << " es la seña" << endl;
				else
					cout << "imagen: " << fileName << " no es la seña" << endl;
			}

		}
	}

};

#endif /* TRAININGMANAGER_H_ */
