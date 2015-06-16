/*
 * SVMClassificator.cpp
 *
 *  Created on: Jun 6, 2013
 *      Author: EPineiro
 */

#include "SVMClassifier.h"

using namespace std;
using namespace cv;

SVMClassifier::SVMClassifier() {
	// TODO Auto-generated constructor stub

}

SVMClassifier::~SVMClassifier() {

	SVM.clear();
}

void SVMClassifier::loadClassifier(const string &file) {

	SVM.clear();
	SVM.load(file.c_str());
	loadScaleFactors(scaleFactorsFile);
}

void SVMClassifier::train(Mat &trainingMat, const Mat &labels, const string svmFileName) {

	findScaleFactors(trainingMat);
	saveScaleFactors(scaleFactorsFile);
	scaleInputUsingIndividualFactors(trainingMat);

	cout<<"training mat despues de escalar: "<<trainingMat<<endl;

	// Setear parametros del SVM
	//CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_EPS, 2000, FLT_EPSILON);
	//CvSVMParams params = CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 10, 1.0, 1000, 0.5, 0.1, NULL, criteria);
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.C = 256;
	params.gamma = 100;

	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, FLT_EPSILON);

	// Entrenar el SVM
	cout << "training the SVM" << endl;
	SVM.train_auto(trainingMat, labels, Mat(), Mat(), params);
	cout << "training terminated" << endl;

	SVM.save(svmFileName.c_str());
}

float SVMClassifier::predict(const vector<float> &input, string inputName) {

	Mat inputMat(1, featuresCount, CV_32FC1);

	int i = 0;
	for(vector<float>::const_iterator it = input.begin(); it != input.end(); it++) {

		inputMat.at<float>(0, i++) = *it;
	}

	scaleInputUsingIndividualFactors(inputMat);

	return SVM.predict(inputMat);
}

float SVMClassifier::getPredictionConfidenceMargin(const vector<float> &input) {

	Mat inputMat(1, featuresCount, CV_32FC1);

	int i = 0;
	for (vector<float>::const_iterator it = input.begin(); it != input.end(); it++) {

		inputMat.at<float> (0, i++) = *it;
	}

	scaleInputUsingIndividualFactors(inputMat);

	return SVM.predict(inputMat, true);
}

