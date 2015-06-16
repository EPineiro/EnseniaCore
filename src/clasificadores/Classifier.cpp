/*
 * Classificator.cpp
 *
 *  Created on: Jun 6, 2013
 *      Author: EPineiro
 */

#include "Classifier.h"
#include "SVMClassifier.h"

using namespace cv;
using namespace std;

Classifier::Classifier() {
	// TODO Auto-generated constructor stub

}

Classifier::~Classifier() {

}

Classifier* Classifier::createClassifier(int type, map<string, string> params) {

	switch(type) {
		case SVM_CLASSIFIER: {
			string file = params["scaleFactorsFile"];
			return new SVMClassifier(file);
			break;
		}
			break;
		default:
			return NULL;
	}
}

void Classifier::findScaleFactors(const Mat &trainingMat) {

	minScalingFactors.resize(featuresCount, 9999999.0);
	maxScalingFactors.resize(featuresCount, -999999.0);

	for(int i = 0; i < trainingMat.rows; i++) {
		for(int j = 0; j < trainingMat.cols; j++) {

			if(trainingMat.at<float>(i, j) < minScalingFactors[j]) {

				minScalingFactors[j] = trainingMat.at<float>(i, j);
			}
			if(trainingMat.at<float>(i, j) > maxScalingFactors[j]) {

				maxScalingFactors[j] = trainingMat.at<float>(i, j);
			}
		}
	}
}

void Classifier::saveScaleFactors(const string &fileName) {

	ofstream writer;
	writer.open(fileName.c_str());
	copy(minScalingFactors.begin(), minScalingFactors.end(), ostream_iterator<float>(writer, ","));
	writer<<endl;
	copy(maxScalingFactors.begin(), maxScalingFactors.end(), ostream_iterator<float>(writer, ","));
	writer.close();
}

void Classifier::loadScaleFactors(const string &fileName) {

	string line;
	ifstream reader(fileName.c_str());
	if(reader.is_open()) {

		string token;

		getline(reader, line);
		stringstream ss(line);
		while (getline(ss, token, ',')) {
			minScalingFactors.push_back(atof(token.c_str()));
		}

		getline(reader, line);
		stringstream ss2(line);
		while (getline(ss2, token, ',')) {
			maxScalingFactors.push_back(atof(token.c_str()));
		}

		reader.close();
	}
}

void Classifier::scaleInputUsingIndividualFactors(Mat &input) {

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {

			input.at<float>(i, j) = (input.at<float>(i, j) - minScalingFactors[j]) / (maxScalingFactors[j] - minScalingFactors[j]);
		}
	}

}

void Classifier::scaleInputUsingExtremeFactors(Mat &input) {

	float minFactor = *min_element(minScalingFactors.begin(), minScalingFactors.end());
	float maxFactor = *max_element(maxScalingFactors.begin(), maxScalingFactors.end());

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {

			input.at<float> (i, j) = (input.at<float> (i, j) - minFactor) / (maxFactor - minFactor);
		}
	}
}
