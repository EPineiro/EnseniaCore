/*
 * DllMain.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: EPineiro
 */

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <iterator>

#include "config/SignCode.h"
#include "features/ImageProcessor.h"
#include "features/SignFeaturesDetector.h"
#include "clasificadores/Classifier.h"
#include "clasificadores/SVMClassifier.h"
#include "clasificadores/RecognitionChain.h"
#include "clasificadores/RecognitionChainNode.h"

using namespace std;
using namespace cv;

extern "C" {

string getSignName(int code);
int getSignCode(string name);

string getLessonName(int code);

VideoCapture *cap;
RecognitionChain *chain;

bool debugEnabled;
string lesson;

__declspec(dllexport) int __cdecl sum(int a, int b) {

	return a + b;
}

__declspec(dllexport) void __cdecl enableDebug() {
	cvNamedWindow("debug", CV_WINDOW_AUTOSIZE);
	debugEnabled = true;
}

__declspec(dllexport) void __cdecl disableDebug() {
	debugEnabled = false;
	cvDestroyWindow("debug");
}

__declspec(dllexport) int __cdecl initCapture(int device) {

	cap = new VideoCapture(device);

	return 0;
}

__declspec(dllexport) int __cdecl endCapture() {

	delete cap;

	return 0;
}

__declspec(dllexport) Mat* __cdecl getFrameFromCamera(unsigned char** buffer, int64* bufferSize, int* height, int* width) {

	if (!cap->isOpened()) {
		cout << "Failed to initialize camera\n";
	}

	Mat frame;
	cap->read(frame);

	Mat* result;
	// Devolver Imagen
	result = new Mat(frame); // Crear la clase MAT
	// Devolver tamaño de imagen
	*height = result->rows;
	*width = result->cols;
	// Copiar imagen binaria
	*buffer = result->data;
	// Calcular el tamaño de la imagen
	*bufferSize = result->dataend - result->datastart;
	// Devolver el puntero a clase MAT
	return result;
}

__declspec(dllexport) void __cdecl deleteFrame(Mat* frame) {

	delete frame;
}

__declspec(dllexport) void __cdecl setLesson(int lessonCode) {

	lesson = getLessonName(lessonCode);
}

__declspec(dllexport) int __cdecl recognize(int signCode, Mat* frame) {

	/*ImageProcessor processor;
	Mat newImage = (*frame).clone();

	newImage = processor.getBlurImage(newImage);
	newImage = processor.filterColorBlobs(newImage);
	newImage = processor.getClosedImage(newImage);

	namedWindow("binary");
	imshow("binary", newImage);
	*/

	string signName = getSignName(signCode);
	stringstream classifierFile;
	classifierFile << "resources/svms_entrenados/" << lesson << "/clasificador_" << signName << ".xml";

	stringstream scaleFactorsFile;
	scaleFactorsFile << "resources/svms_entrenados/" << lesson << "/svm_scale_factors.txt";
	map<string, string> params;
	params["scaleFactorsFile"] = scaleFactorsFile.str();

	Classifier *classifier = Classifier::createClassifier(Classifier::SVM_CLASSIFIER, params);
	classifier->loadClassifier(classifierFile.str());
	classifier->setFeaturesCount(12);

	SignFeaturesDetector detector;

	if (debugEnabled) detector.setDebugMode(true);
	vector<float> features = detector.getSignFeatures(*frame, SignFeaturesDetector::USE_BLOBS);

	float response = classifier->predict(features);

	delete classifier;

	if (fabs(response - 1.0) <= FLT_EPSILON) {
		return 1;
	} else {
		return 0;
	}
}

__declspec(dllexport) int __cdecl initRecognitionChain() {

	DIR *dp;
	struct dirent *dirp;

	string dir = "resources/svms_entrenados/alfabeto";

	chain = new RecognitionChain();
	chain->setNotMatchingCode(-1);
	chain->loadErrorsMargins("resources/sign_error_margins.xml");

	if ((dp = opendir(dir.c_str())) == NULL) {
		cout << "Error abriendo directorio con svm's entrenados" << endl;
	}

	while ((dirp = readdir(dp)) != NULL) {

		string fileName = string(dirp->d_name);

		if (fileName.find("clasificador") != string::npos) {

			string sign = fileName.substr(fileName.find_first_of('_') + 1, fileName.find_first_of('.') - fileName.find_first_of('_') - 1);

			stringstream classifierFile;
			classifierFile << dir << "/" << fileName;

			map<string, string> params;
			params["scaleFactorsFile"] = "resources/svms_entrenados/alfabeto/svm_scale_factors.txt";

			Classifier *classifier = Classifier::createClassifier(Classifier::SVM_CLASSIFIER, params);
			classifier->loadClassifier(classifierFile.str());
			classifier->setFeaturesCount(12);
			RecognitionChainNode node(getSignCode(sign), classifier);
			chain->addNode(node);

		}
	}

	closedir(dp);

	return 0;
}

__declspec(dllexport) int __cdecl deleteRecognitionChain() {

	delete chain;

	return 0;
}

__declspec(dllexport) int __cdecl recognizeAnySign(Mat* frame) {

	SignFeaturesDetector detector;

	vector<float> features = detector.getSignFeatures(*frame, SignFeaturesDetector::USE_BLOBS);

	return chain->recognize(features);
}

string getSignName(int code) {

	string fileName = "resources/sign_configs.xml";

	FileStorage fs;
	fs.open(fileName, FileStorage::READ);

	FileNode node = fs["Signs"];

	for (FileNodeIterator it = node.begin(); it != node.end(); it++) {

		SignCode sign;
		(*it) >> sign;

		if(sign.code == code)
			return sign.name;
	}

	return "";
}

string getLessonName(int code) {

	string fileName = "resources/lessons_config.xml";

	FileStorage fs;
	fs.open(fileName, FileStorage::READ);

	FileNode node = fs["Lessons"];

	for (FileNodeIterator it = node.begin(); it != node.end(); it++) {

		SignCode sign;
		(*it) >> sign;

		if(sign.code == code)
			return sign.name;
	}

	return "";
}

int getSignCode(string name) {

	string fileName = "resources/sign_configs.xml";

	FileStorage fs;
	fs.open(fileName, FileStorage::READ);

	FileNode node = fs["Signs"];

	for (FileNodeIterator it = node.begin(); it != node.end(); it++) {

		SignCode sign;
		(*it) >> sign;

		if (sign.name == name)
			return sign.code;
	}

	return -1;
}

}
