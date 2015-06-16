/*
 * SignFeaturesDetector.h
 *
 *  Created on: Jul 7, 2013
 *      Author: EPineiro
 */

#ifndef SIGNFEATURESDETECTOR_H_
#define SIGNFEATURESDETECTOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>

#include "../cvBlob/cvblob.h"

#include "../features/ImageProcessor.h"
#include "../features/ImageGeometricFeatureDetector.h"

class SignFeaturesDetector {
private:

	std::vector<cv::Point> handTrajectory;

	std::vector<float> getSignFeaturesUsingBlobs(cv::Mat image, std::string name = "");
	std::vector<float> getSignFeaturesUsingInvariantFeatures(cv::Mat image, std::string name = "");
	std::vector<float> getSignFeaturesUsingHuMoments(cv::Mat image, std::string name = "");

	bool debugEnabled;
	int getHandPositionToFace(cv::Rect handPos, cv::Rect facePos);

public:

	enum {
			USE_INVARIANT_FEATURES,
			USE_BLOBS,
			USE_HU_MOMENTS
		};

	std::vector<float> getSignFeatures(cv::Mat image, int featuresType, std::string name = "");
	void setDebugMode(bool mode);

	SignFeaturesDetector();
	virtual ~SignFeaturesDetector();
};

#endif /* SIGNFEATURESDETECTOR_H_ */
