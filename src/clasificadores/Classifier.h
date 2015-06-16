/*
 * Classificator.h
 *
 *  Created on: Jun 6, 2013
 *      Author: EPineiro
 */

#ifndef CLASSIFICATOR_H_
#define CLASSIFICATOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>

class Classifier {

protected:
	int featuresCount;
	std::string file;

	std::vector<float> minScalingFactors;
	std::vector<float> maxScalingFactors;

	void findScaleFactors(const cv::Mat &trainingMat);
	void saveScaleFactors(const std::string &fileName);
	void loadScaleFactors(const std::string &fileName);
public:

	enum {
		SVM_CLASSIFIER,
		NAIVE_BAYES_CLASSIFIER,
		RANDOM_FOREST_CLASSIFIER,
		K_NEAREST_CLASSIFIER
	};

	Classifier();
	virtual ~Classifier();

	inline void setFeaturesCount(int featuresCount) {
		this->featuresCount = featuresCount;
	}

	static Classifier* createClassifier(int type, std::map<std::string, std::string> params);
	virtual void loadClassifier(const std::string &file) = 0;

	void scaleInputUsingIndividualFactors(cv::Mat &input);
	void scaleInputUsingExtremeFactors(cv::Mat &input);

	virtual void train(cv::Mat &trainingMat, const cv::Mat &labels, const std::string fileName) = 0;
	virtual float predict(const std::vector<float> &input, std::string inputName = "") = 0;

	virtual float getPredictionConfidenceMargin(const std::vector<float> &input) = 0;
};

#endif /* CLASSIFICATOR_H_ */
