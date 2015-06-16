/*
 * ImageProcessor.h
 *
 *  Created on: May 27, 2013
 *      Author: EPineiro
 */

#ifndef IMAGEPROCESSOR_H_
#define IMAGEPROCESSOR_H_

#include <opencv/cv.h>

class ImageProcessor {
public:
	ImageProcessor();
	virtual ~ImageProcessor();

	cv::Mat getBlurImage(const cv::Mat &image);
	cv::Mat getGrayscaleForm(const cv::Mat &image);
	cv::Mat getBinaryForm(const cv::Mat &image);
	cv::Mat getClosedImage(const cv::Mat &image);
	cv::Mat getOpenedImage(const cv::Mat &image);
	cv::Mat filterSkinTone(const cv::Mat &image);
	cv::Mat filterColorBlobs(const cv::Mat &image);

	cv::Mat augmentContrast(const cv::Mat &image);
	cv::Mat augmentBrightness(const cv::Mat &image);
};

#endif /* IMAGEPROCESSOR_H_ */
