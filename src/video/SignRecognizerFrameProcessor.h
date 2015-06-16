/*
 * SignRecognizerFrameProcessor.h
 *
 *  Created on: May 29, 2013
 *      Author: EPineiro
 */

#ifndef SIGNRECOGNIZERFRAMEPROCESSOR_H_
#define SIGNRECOGNIZERFRAMEPROCESSOR_H_

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iterator>

#include "FrameProcessor.h"
#include "../clasificadores/Classifier.h"
#include "../clasificadores/RecognitionChain.h"

class SignRecognizerFrameProcessor: public FrameProcessor {
private:
	cv::Rect roi;
	int featuresType;
	RecognitionChain recognizer;

	void addImageToSVMMat(cv::Mat trainingMat, int index, cv::Mat image);

public:
	SignRecognizerFrameProcessor();
	virtual ~SignRecognizerFrameProcessor();

	inline void setRoi(cv::Rect roi) {

		this->roi = roi;
	}

	inline void setFeaturesType(int type) {
		this->featuresType = type;
	}

	inline void setRecognizer(RecognitionChain &chain) {
		this->recognizer = chain;
	}

	void drawMessage(const std::string text, cv::Mat &output);

	virtual void process(cv::Mat &input, cv::Mat &output);

};

#endif /* SIGNRECOGNIZERFRAMEPROCESSOR_H_ */
