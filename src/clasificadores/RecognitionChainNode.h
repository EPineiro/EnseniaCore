/*
 * RecognitionChainNode.h
 *
 *  Created on: Jun 22, 2013
 *      Author: EPineiro
 */

#ifndef RECOGNITIONCHAINNODE_H_
#define RECOGNITIONCHAINNODE_H_

#include <opencv2/opencv.hpp>
#include <stdlib.h>

#include "../features/SignFeaturesDetector.h"

using namespace std;

class RecognitionChainNode {

private:
	Classifier *classifier;
	int signCode;
	float errorMargin;

public:
	RecognitionChainNode() {

	}

	RecognitionChainNode(const int signCode, Classifier* classifier) {

		this->classifier = classifier;
		this->signCode = signCode;
	}

	virtual ~RecognitionChainNode(){

	}

	inline int getSignCode() {
		return this->signCode;
	}

	inline float getErrorMargin() {
		return this->errorMargin;
	}

	bool recognize(const vector<float> &features) {

		float response = classifier->predict(features);
		errorMargin = classifier->getPredictionConfidenceMargin(features);

		if (fabs(response - 1.0) <= FLT_EPSILON) {

			return true;
		} else {

			return false;
		}
	}
};

#endif /* RECOGNITIONCHAINNODE_H_ */
