/*
 * SignRecognizerFrameProcessor.cpp
 *
 *  Created on: May 29, 2013
 *      Author: EPineiro
 */

#include "SignRecognizerFrameProcessor.h"

#include "../clasificadores/SVMClassifier.h"

using namespace cv;
using namespace std;

SignRecognizerFrameProcessor::SignRecognizerFrameProcessor() {
	// TODO Auto-generated constructor stub

}

SignRecognizerFrameProcessor::~SignRecognizerFrameProcessor() {

}

void SignRecognizerFrameProcessor::drawMessage(const string text, Mat &output) {

	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 2;
	int thickness = 3;

	putText(output, text, Point(500, 300), fontFace, fontScale,	Scalar::all(255), thickness, 8);
}

void SignRecognizerFrameProcessor::process(Mat &input, Mat &output) {

	output = input.clone();

	SignFeaturesDetector detector;

	vector<float> features = detector.getSignFeatures(input, featuresType);

	//drawMessage(recognizer.recognize(features), output);
}
