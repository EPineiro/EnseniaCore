/*
 * SignFeaturesDetector.cpp
 *
 *  Created on: Jul 7, 2013
 *      Author: EPineiro
 */

#include "SignFeaturesDetector.h"

using namespace cv;
using namespace cvb;
using namespace std;

SignFeaturesDetector::SignFeaturesDetector() {
	// TODO Auto-generated constructor stub
	debugEnabled = false;
}

SignFeaturesDetector::~SignFeaturesDetector() {
	// TODO Auto-generated destructor stub
}

vector<float> SignFeaturesDetector::getSignFeatures(Mat image, int featuresType, string name) {

	switch(featuresType) {
		case USE_BLOBS:
			return getSignFeaturesUsingBlobs(image, name);
		default:
			return vector<float>();
	}
}

vector<float> SignFeaturesDetector::getSignFeaturesUsingBlobs(Mat image, string name) {

	vector<float> features;
	Rect roi(35, 170, 210, 220);
	Mat imageRoi = image(roi);

	Mat output = image.clone();

	ImageProcessor processor;
	ImageGeometricFeatureDetector detector;

	Mat newImage = image.clone();

	newImage = processor.getBlurImage(newImage);
	newImage = processor.filterColorBlobs(newImage);
	newImage = processor.getClosedImage(newImage);

	// MOSTRAR DEBUG
	if (debugEnabled) imshow("debug", newImage);

	CvBlobs blobs = detector.getImageBlobs(newImage);
	CvBlobs blobsFiltereds = detector.getHandsBlobs(blobs);

	if(!blobsFiltereds.empty()) {

		//dibujar blobs
		/*
		IplImage img = imageRoi;
		IplImage *labelImg = cvCreateImage(cvSize(newImage.cols, newImage.rows), IPL_DEPTH_LABEL, 1);
		cvRenderBlobs(labelImg, blobsFiltereds, &img, &img);
		*/

/*		for (CvBlobs::const_iterator it = blobsFiltereds.begin(); it != blobsFiltereds.end(); ++it) {

			switch(it->first) {
				case ImageGeometricFeatureDetector::FACE_BLOB:
					putText(newImage, "cara", Point(it->second->minx, it->second->miny), FONT_HERSHEY_SCRIPT_SIMPLEX, 2,	Scalar::all(255), 3, 8);
					break;
				case ImageGeometricFeatureDetector::RIGHT_HAND_BLOB:
					putText(newImage, "derecha", Point(50, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2,	Scalar::all(255), 3, 8);
					break;
				case ImageGeometricFeatureDetector::LEFT_HAND_BLOB:
					putText(newImage, "izquierda", Point(500, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2,	Scalar::all(255), 3, 8);
					break;
			}
		}
*/

		//sacamos los features de una de las manos
		CvBlobs::iterator it2 = blobsFiltereds.begin();

		if(it2 != blobsFiltereds.end()) {

			CvBlob* handBlob = it2->second;

			CvContourPolygon* countour = cvConvertChainCodesToPolygon(&handBlob->contour);
			vector<Point> convertedContour = detector.convertToVectorOfPoints(*countour);

			vector<Point> aproxContour = detector.aproxContours(convertedContour, ImageGeometricFeatureDetector::POLY_APROX);
			vector<Point> convexHull = detector.aproxContours(convertedContour, ImageGeometricFeatureDetector::CONVEX_HULL_APROX);
			//detector.drawContour(output, aproxContour);

			Rect boundingBox(handBlob->minx, handBlob->miny, handBlob->maxx - handBlob->minx, handBlob->maxy - handBlob->miny);

			double hus[7];
			Moments moment = moments(aproxContour, false);
			HuMoments(moment, hus);
			for (int i = 0; i < 7; i++) {
				features.push_back(hus[i]);
			}

			features.push_back(handBlob->area / contourArea(Mat(convexHull))); //solidez
			features.push_back(pow(arcLength(Mat(aproxContour), true), 2) / contourArea(Mat(aproxContour))); //redondez
			features.push_back(contourArea(Mat(aproxContour)) / (boundingBox.width * boundingBox.height)); //rectangularidad
			//triangularidad
			double I = ((handBlob->u20 * handBlob->u02) - pow(handBlob->u11, 2)) / pow(handBlob->m00, 4);
			double triangularity = (I <= 1 / 108) ? (108 * I) : (1 / (108 * I));
			features.push_back(triangularity);

			//posicion de las manos respecto a la cara
			Rect facePosition = detector.getFacePosition(image);
			Rect handPosition = boundingBox;
			features.push_back(getHandPositionToFace(handPosition, facePosition));

		/*	switch (getHandPositionToFace(handPosition, facePosition)) {
			case ImageGeometricFeatureDetector::MERGED:
				putText(newImage, "Mezcladas", Point(50, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::UP:
				putText(newImage, "Arriba", Point(50, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::RIGHT:
				putText(newImage, "derecha", Point(50, 100), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::LEFT:
				putText(newImage, "izquierda", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::BOTTOM:
				putText(newImage, "abajo", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::UP_RIGHT:
				putText(newImage, "arriba-derecha", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::UP_LEFT:
				putText(newImage, "arriba-izquierda", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::BOTTOM_RIGHT:
				putText(newImage, "abajo-derecha", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			case ImageGeometricFeatureDetector::BOTTOM_LEFT:
				putText(newImage, "abajo-izquierda", Point(50, 100),	FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
				break;
			}
		*/

			//agregamos un nuevo punto a la trayectoria de la mano solo si no se encuentra demasiado alejado del ultimo
			//(lo que podria significar que solo es ruido)
			Point newPoint(handBlob->centroid.x, handBlob->centroid.y);
			//el primer punto lo agregamos siempre
			if (handTrajectory.empty())
				handTrajectory.push_back(newPoint);

			if (detector.euclideanDist(newPoint, handTrajectory.back()) < 50)
				handTrajectory.push_back(newPoint);

			//dibujamos la trayectoria
			for (vector<Point>::const_iterator it = handTrajectory.begin() + 1; it != handTrajectory.end(); it++) {

				line(output, *(it - 1), *it, Scalar(0), 2);
			}
		}
	}

	cvReleaseBlobs(blobs);


	//namedWindow(name);
	//imshow(name, output);

	//
	std::stringstream binaryName;
	binaryName<<name<<"-binary.jpg";
	//namedWindow(binaryName.str());
	//imshow(binaryName.str(), newImage);
	//imwrite(binaryName.str(), newImage);
 	//*/
	return features;
}

int SignFeaturesDetector::getHandPositionToFace(Rect handPos, Rect facePos) {

	Rect intersection = handPos & facePos;
	//si la interseccion no es nula, entonces las manos se mezclan con la cara
	if(intersection != Rect())
		return ImageGeometricFeatureDetector::MERGED;


	if(handPos.x < facePos.x && abs((int)(handPos.x - facePos.x)) > 50) {

		if(abs((int)(handPos.y - facePos.y)) < 100)
			return ImageGeometricFeatureDetector::RIGHT;
		else if(handPos.y < facePos.y)
			return ImageGeometricFeatureDetector::UP_RIGHT;
		else if(handPos.y > facePos.y)
			return ImageGeometricFeatureDetector::BOTTOM_RIGHT;
	}
	else if(handPos.x > facePos.x && abs((int)(handPos.x - facePos.x)) > 50) {

		if(abs((int)(handPos.y - facePos.y)) < 100)
			return ImageGeometricFeatureDetector::LEFT;
		else if(handPos.y < facePos.y)
			return ImageGeometricFeatureDetector::UP_LEFT;
		else if(handPos.y > facePos.y)
			return ImageGeometricFeatureDetector::BOTTOM_LEFT;
	}
	else if (handPos.y < facePos.y && abs((int)(handPos.y - facePos.y)) > 50) {

		if (abs((int)(handPos.x - facePos.x)) < 100)
			return ImageGeometricFeatureDetector::UP;
		else if (handPos.x < facePos.x)
			return ImageGeometricFeatureDetector::UP_RIGHT;
		else if (handPos.x > facePos.x)
			return ImageGeometricFeatureDetector::UP_LEFT;
	}
	else if (handPos.y > facePos.y && abs((int)(handPos.y - facePos.y)) > 50) {

		if (abs((int)(handPos.x - facePos.x)) < 100)
			return ImageGeometricFeatureDetector::BOTTOM;
		else if (handPos.x < facePos.x)
			return ImageGeometricFeatureDetector::BOTTOM_RIGHT;
		else if (handPos.x > facePos.x)
			return ImageGeometricFeatureDetector::BOTTOM_LEFT;
	}
}

void SignFeaturesDetector::setDebugMode(bool mode) {
	debugEnabled = mode;
}
