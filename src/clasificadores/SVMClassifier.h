/*
 * SVMClassificator.h
 *
 *  Created on: Jun 6, 2013
 *      Author: EPineiro
 */
#ifndef SVMCLASSIFICATOR_H_
#define SVMCLASSIFICATOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "Classifier.h"

class SVMClassifier: public Classifier {
private:
	CvSVM SVM;
	std::string scaleFactorsFile;
public:
	SVMClassifier();
	SVMClassifier(std::string _scaleFactorsFile): scaleFactorsFile(_scaleFactorsFile) {}
	virtual ~SVMClassifier();

	virtual void loadClassifier(const std::string &file);

	virtual void train(cv::Mat &trainingMat, const cv::Mat &labels, const std::string svmFileName);
	virtual float predict(const std::vector<float> &input, std::string inputName = "");

	virtual float getPredictionConfidenceMargin(const std::vector<float> &input);
};

#endif /* SVMCLASSIFICATOR_H_ */
