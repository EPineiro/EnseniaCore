/*
 * ImageFeatureDetector.h
 *
 *  Created on: May 1, 2013
 *      Author: EPineiro
 */

#ifndef IMAGEFEATUREDETECTOR_H_
#define IMAGEFEATUREDETECTOR_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <algorithm>

#include "../cvBlob/cvblob.h"

class ImageGeometricFeatureDetector {
private:
	void prepareImageForContourDetection(cv::Mat &image);

public:

	enum {
		POLY_APROX,
		CONVEX_HULL_APROX
	};

	enum {
		FACE_BLOB,
		RIGHT_HAND_BLOB,
		LEFT_HAND_BLOB
	};

	enum {
		UP,
		UP_RIGHT,
		RIGHT,
		BOTTOM_RIGHT,
		BOTTOM,
		BOTTOM_LEFT,
		LEFT,
		UP_LEFT,
		MERGED
	};

	inline double euclideanDist(cv::Point p, cv::Point q) {
		    cv::Point diff = p - q;
		    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
		}

	ImageGeometricFeatureDetector();
	virtual ~ImageGeometricFeatureDetector();

	cv::Mat getImageEdges(const cv::Mat &image);
	std::vector<cv::Vec3f> getCircles(const cv::Mat &image);
	std::vector<std::vector<cv::Point> > findComponentsContours(const cv::Mat &image, bool preprocess = false);
	std::vector<cv::Point> aproxContours(const std::vector<cv::Point> &contour, int aproxType = POLY_APROX);
	std::vector<cv::Point> getMaxAreaContour(const std::vector<std::vector<cv::Point> > &contours);
	std::vector<cv::Vec4i> getConvexityDefects(const std::vector<cv::Point> &contour);
	std::vector<cv::Point> findFarthestPointsFromCentroid(const std::vector<cv::Point> &contour, const cv::Point &centroid, const unsigned int cantPoints);
	std::vector<cv::Point> findIntersectionBetweenContourAndConvexHull(const std::vector<cv::Point> &contour, const std::vector<cv::Point> &convexHull);
	std::vector<double> getDistancesToCentroid(const std::vector<cv::Point> &contour, const cv::Point &centroid);

	cvb::CvBlobs getImageBlobs(const cv::Mat &image);
	cvb::CvBlobs getHandsBlobs(const cvb::CvBlobs &blobs);

	cv::Rect getFacePosition(const cv::Mat &image);

	cv::Point scaleSinglePointToSquare(const cv::Point &point, const cv::Rect &boundingBox, const double squareSize);
	std::vector<cv::Point> scalePointsToSquare(const std::vector<cv::Point> &points, const cv::Rect &boundingBox, const double squareSize);
	std::vector<cv::Point> translatePointsToCentroidAsOrigin(const std::vector<cv::Point> &points, const cv::Point &centroid);
	void sortPointsByXcoordinate(std::vector<cv::Point> &points);

	void drawCircles(cv::Mat &image, const std::vector<cv::Vec3f> &circles);
	void drawContour(cv::Mat &image, const std::vector<cv::Point> &contour);
	void drawConvexityDefects(cv::Mat &image, std::vector<cv::Vec4i> &defects, std::vector<cv::Point> &contour);
	void drawPoints(cv::Mat &image, std::vector<cv::Point> &points);

	std::vector<cv::Point> convertToVectorOfPoints(const cvb::CvContourPolygon &contour);
};

#endif /* IMAGEFEATUREDETECTOR_H_ */
