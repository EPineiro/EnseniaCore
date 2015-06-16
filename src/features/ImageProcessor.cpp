/*
 * ImageProcessor.cpp
 *
 *  Created on: May 27, 2013
 *      Author: EPineiro
 */

#include "ImageProcessor.h"

using namespace cv;

ImageProcessor::ImageProcessor() {
	// TODO Auto-generated constructor stub

}

ImageProcessor::~ImageProcessor() {
	// TODO Auto-generated destructor stub
}

Mat ImageProcessor::getBlurImage(const Mat &image) {

	Mat newImage = image.clone();
	GaussianBlur(image, image, Size(5,5), 1.5);
	return newImage;
}

Mat ImageProcessor::getGrayscaleForm(const Mat &image) {

	Mat newImage = image.clone();
	cvtColor(image, newImage, CV_BGR2GRAY);
	return newImage;
}

Mat ImageProcessor::getBinaryForm(const Mat &image) {

	Mat newImage = image.clone();
	threshold(image, newImage, 125, 255, THRESH_BINARY_INV);
	return newImage;
}

Mat ImageProcessor::getClosedImage(const Mat &image) {

	Mat newImage = image.clone();
	//Mat kernel(7, 7, CV_8U, Scalar(1));
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11,11));
	morphologyEx(image, newImage, MORPH_CLOSE, kernel, Point(-1, -1), 1);
	return newImage;
}

Mat ImageProcessor::getOpenedImage(const Mat &image) {

	Mat newImage = image.clone();
	Mat kernel(5, 5, CV_8U, Scalar(1));
	morphologyEx(image, newImage, MORPH_OPEN, kernel);
	return newImage;
}

Mat ImageProcessor::filterSkinTone(const Mat &image) {

	Mat newImage = image.clone();
	GaussianBlur(image, newImage, Size(5,5), 1.5);
/*	cvtColor(newImage, newImage, CV_BGR2HSV);
	 //la convertimos en una imagen binaria, dejando solo el tono de piel humana
	 inRange(newImage, Scalar(0, 10, 60), Scalar(20, 150, 255), newImage);
*/
	cvtColor(newImage, newImage, CV_BGR2YCrCb);
	//la convertimos en una imagen binaria, dejando solo el tono de piel humana
	inRange(newImage, Scalar(0, 135, 80), Scalar(255, 200, 140), newImage);

//	cvtColor(newImage, newImage, CV_BGR2HSV);
//	inRange(newImage, Scalar(74, 0, 0), Scalar(125, 255, 164), newImage);

	//GaussianBlur(newImage, newImage, Size(5,5), 1.5);

	return newImage;
}

Mat ImageProcessor::filterColorBlobs(const Mat &image) {

	Mat newImage = image.clone();
	GaussianBlur(image, newImage, Size(5,5), 1.5);

	cvtColor(newImage, newImage, CV_BGR2HSV);
	//azul oscuro
	//inRange(newImage, Scalar(112, 102, 0), Scalar(127, 255, 164), newImage);
	//celeste
	inRange(newImage, Scalar(98, 48, 41), Scalar(138, 255, 256), newImage);

	return newImage;
}

Mat ImageProcessor::augmentContrast(const Mat &image) {

	Mat newImage = image.clone();
	for (int i = 0; i < newImage.rows; i++) {
		for (int j = 0; j < newImage.cols; j++) {
			for (int k = 0; k < 3; k++) {

				newImage.at<Vec3b> (i, j)[k] = saturate_cast<uchar> (newImage.at<Vec3b> (i, j)[k] * 2);
			}
		}
	}

	return newImage;
}

Mat ImageProcessor::augmentBrightness(const Mat &image) {

	Mat newImage = image.clone();
	for (int i = 0; i < newImage.rows; i++) {
		for (int j = 0; j < newImage.cols; j++) {
			for (int k = 0; k < 3; k++) {

				newImage.at<Vec3b> (i, j)[k] = saturate_cast<uchar> (newImage.at<Vec3b> (i, j)[k] + 50);
			}
		}
	}

	return newImage;
}
