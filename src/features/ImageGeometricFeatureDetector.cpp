/*
 * ImageFeatureDetector.cpp
 *
 *  Created on: May 1, 2013
 *      Author: EPineiro
 */

#include "ImageGeometricFeatureDetector.h"

using namespace cv;
using namespace cvb;
using namespace std;

bool comparePointsByXcoordinateFunc(Point p1, Point p2) {

	return p1.x < p2.x;
}

ImageGeometricFeatureDetector::ImageGeometricFeatureDetector() {
	// TODO Auto-generated constructor stub

}

ImageGeometricFeatureDetector::~ImageGeometricFeatureDetector() {
	// TODO Auto-generated destructor stub
}

Mat ImageGeometricFeatureDetector::getImageEdges(const Mat &image) {

	int lowThreshold = 80;
	Mat edges = image.clone();

	//cvtColor( edges, edges, CV_BGR2GRAY );
	medianBlur( edges, edges, 7 );

	/*/
	vector<Mat> channels;
	cvtColor(edges, edges, CV_BGR2HSV);
	split(edges, channels);
	edges = channels[0];
	//*/
	//algoritmo de Canny para detectar bordes
	Canny(edges, edges, lowThreshold, lowThreshold * 3);

	//morphologyEx(edges, edges, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1), 1);
	morphologyEx(edges, edges, MORPH_GRADIENT, getStructuringElement(MORPH_ELLIPSE, Size(5,5)), Point(-1,-1), 1);
	morphologyEx(edges, edges, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3,3)), Point(-1,-1), 1);


	//dilate(edges, edges, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

	//GaussianBlur( edges, edges, Size(3,3), 0, 0, BORDER_DEFAULT );

	//para invertir los colores y tener los bordes en negro
	threshold(edges, edges, 0, 255, THRESH_BINARY_INV);

	/*
	GaussianBlur( edges, edges, Size(3,3), 0, 0, BORDER_DEFAULT );
	cvtColor( edges, edges, CV_BGR2GRAY );
	Laplacian(edges, edges, CV_16S);
	convertScaleAbs(edges, edges);
	*/

	return edges;
}

/**
 * Devuelve los circulos encontrados en la imagen.
 * LA IMAGEN DEBE SER EN ESCALA DE GRISES (8 bits, un solo canal)
 */
vector<Vec3f> ImageGeometricFeatureDetector::getCircles(const Mat &image) {

	GaussianBlur(image, image, Size(5, 5), 1.5);
	std::vector<Vec3f> circles;

	HoughCircles(image, circles, CV_HOUGH_GRADIENT, 2, //resolucion del acumulador (tamaño de la imagen dividido 2
			50, //distancia minima entre circulos
			200, //threshold maximo para la parte del algoritmo de Canny (llamado dentro de esta funcion)
			100, //numero minimo de votos para una linea
			25, 100 //radios minimo y maximo para un circulo
	);

	return circles;

}

/**
 * Devuelve los contornos de los componentes en una imagen
 */
vector<std::vector<Point> > ImageGeometricFeatureDetector::findComponentsContours(const Mat &image, bool preprocess) {

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//necesitamos una copia de la imagen porque la original es modificada
	//la imagen a aplicar el algoritmo debe ser binaria
	Mat tmp = image.clone();

	if(preprocess)
		prepareImageForContourDetection(tmp);

	findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	return contours;
}

void ImageGeometricFeatureDetector::prepareImageForContourDetection(Mat &image) {

	//suavizamos la imagen
	GaussianBlur(image, image, Size(5,5), 1.5);
	cvtColor(image, image, CV_BGR2YCrCb);
	//la convertimos en una imagen binaria, dejando solo el tono de piel humana
	inRange(image, Scalar(0, 135, 80), Scalar(255, 200, 140), image);
	//threshold(image, image, 100, 255, THRESH_BINARY);
	//abrimos la imagen para mejorarla un poco mas los contornos
	Mat kernel(5, 5, CV_8U, Scalar(1));
	morphologyEx(image, image, MORPH_OPEN, kernel);
}

/**
 * Aproxima los contornos de un componente de una imagen para obtener la forma del mismo
 * Devuelve un vector con los puntos que determinan los vertices del contorno
 */
vector<Point> ImageGeometricFeatureDetector::aproxContours(const vector<Point> &contour, int aproxType) {

	vector<Point> contourAprox;

	switch (aproxType) {

	case POLY_APROX:
		approxPolyDP(Mat(contour), contourAprox, 5, true);
		break;
	case CONVEX_HULL_APROX:
		convexHull(Mat(contour), contourAprox, true);
		break;
	}

	return contourAprox;
}

vector<Point> ImageGeometricFeatureDetector::getMaxAreaContour(const vector<vector<Point> > &contours) {

	vector<Point> maxContour;
	double area = 0.0, maxArea = 0.0;
	for (vector<vector<Point> >::const_iterator it = contours.begin(); it != contours.end(); it++) {

		area = fabs(contourArea(*it));
		if (area > maxArea) {
			maxArea = area;
			maxContour = *it;
		}
	}

	return maxContour;
}

vector<Vec4i> ImageGeometricFeatureDetector::getConvexityDefects(const vector<Point> &contour) {

	//primero calculamos el convex hull, pero como un vector de indices, en vez de uno de puntos.
	vector<int> hull;
	convexHull(contour, hull, true);

	//ahora los defectos convexos
	vector<Vec4i> defects;
	convexityDefects(contour, hull, defects);

	return defects;
}

vector<Point> ImageGeometricFeatureDetector::findFarthestPointsFromCentroid(const vector<Point> &contour, const Point &centroid, const unsigned int cantPoints) {

	if(contour.size() <= cantPoints)
		return contour;

	vector<Point> farthestPoints, tmp(contour);
	for(unsigned int i = 0; i < cantPoints; i++) {

		double maxDistance = 0;
		vector<Point>::iterator maxPoint;
		for(vector<Point>::iterator it = tmp.begin(); it != tmp.end(); it++) {

			double dist = euclideanDist(*it, centroid);
			if(dist > maxDistance) {
				maxDistance = dist;
				maxPoint = it;
			}
		}

		//si el punto esta muy cerca a otro que ya agregamos lo ignoramos

		farthestPoints.push_back(*maxPoint);
		tmp.erase(maxPoint);
	}

	return farthestPoints;
}

vector<Point> ImageGeometricFeatureDetector::findIntersectionBetweenContourAndConvexHull(const vector<Point> &contour, const vector<Point> &convexHull) {

	vector<Point> farthestPoints, tmp(contour);

	for (vector<Point>::const_iterator it = contour.begin(); it != contour.end(); it++) {

		for (vector<Point>::const_iterator it2 = convexHull.begin(); it2 != convexHull.end(); it2++) {

			if(it->x == it2->x && it->y == it2->y) {
				farthestPoints.push_back(*it);
			}
		}
	}

	return farthestPoints;
}


CvBlobs ImageGeometricFeatureDetector::getImageBlobs(const Mat &image) {

	IplImage gray_img = image;
	IplImage *labelImg = cvCreateImage(cvSize(gray_img.width, gray_img.height),	IPL_DEPTH_LABEL, 1);

	CvBlobs blobs;
	cvLabel(&gray_img, labelImg, blobs);

	cvReleaseImage(&labelImg);

	return blobs;
}

CvBlobs ImageGeometricFeatureDetector::getHandsBlobs(const CvBlobs &blobs) {

	CvBlobs blobsFiltereds;
	CvBlobs result;

	if (!blobs.empty()) {

		//sacamos todos los blobs candidatos, que serian los mas grandes, idealmente deberian ser 3 o menos
		for (CvBlobs::const_iterator it = blobs.begin(); it != blobs.end(); ++it) {

			if ((*it).second->area > 1000) {

				blobsFiltereds.insert(*it);
			}
		}
	}

	return blobsFiltereds;
}

Rect ImageGeometricFeatureDetector::getFacePosition(const Mat &image) {

	//buscamos la cara en la imagen
	CascadeClassifier *face_detector = new CascadeClassifier();
	face_detector->load("resources/otros/lbpcascade_frontalface.xml");
	vector<Rect> face_detections;

	Rect result;

	face_detector->detectMultiScale(image, face_detections);

	if (!face_detections.empty()) {

		result = face_detections.at(0);
	}

	delete face_detector;

	return result;
}

vector<Point> ImageGeometricFeatureDetector::scalePointsToSquare(const vector<Point> &points, const Rect &boundingBox, const double squareSize) {

	vector<Point> scaleds;
	for (vector<Point>::const_iterator it = points.begin(); it != points.end(); it++) {

		scaleds.push_back(scaleSinglePointToSquare(*it, boundingBox, squareSize));
	}

	return scaleds;
}

Point ImageGeometricFeatureDetector::scaleSinglePointToSquare(const Point &point, const Rect &boundingBox, const double squareSize) {

	Point scaled;

	double scaledX = point.x * (squareSize / boundingBox.width);
	double scaledY = point.y * (squareSize / boundingBox.height);

	return Point(scaledX, scaledY);
}

vector<Point> ImageGeometricFeatureDetector::translatePointsToCentroidAsOrigin(const vector<Point> &points, const Point &centroid) {

	vector<Point> result;

	for (vector<Point>::const_iterator it = points.begin(); it != points.end(); it++) {

		double qx = (*it).x - centroid.x;
		double qy = (*it).y - centroid.y;
		result.push_back(Point(qx, qy));
	}

	return result;
}

void ImageGeometricFeatureDetector::sortPointsByXcoordinate(vector<Point> &points) {

	sort(points.begin(), points.end(), comparePointsByXcoordinateFunc);
}

vector<double> ImageGeometricFeatureDetector::getDistancesToCentroid(const vector<Point> &contour, const cv::Point &centroid) {

	vector<double> distances;

	for (vector<Point>::const_iterator it = contour.begin(); it != contour.end(); it++) {

		distances.push_back(euclideanDist(*it, centroid));
	}

	return distances;
}

void ImageGeometricFeatureDetector::drawCircles(Mat &image, const std::vector<Vec3f> &circles) {

	for (std::vector<Vec3f>::const_iterator it = circles.begin(); it != circles.end(); it++) {

		circle(image, Point((*it)[0], (*it)[1]), //centro del circulo
				(*it)[2], //radio del circulo
				Scalar(255), //color,
				2 //grosor del borde
		);
	}
}

void ImageGeometricFeatureDetector::drawContour(Mat &image, const std::vector<Point> &contour) {

	for (std::vector<Point>::const_iterator it = contour.begin(); it != (contour.end() - 1); it++) {

		line(image, *it, *(it + 1), Scalar(0,0,255), 2);
	}

	//conectamos el ultimo punto al primero para cerrar el contorno
	line(image, *(contour.begin()), *(contour.end() - 1), Scalar(0,0,255), 2);
}

void ImageGeometricFeatureDetector::drawConvexityDefects(Mat &image, vector<Vec4i> &defects, vector<Point> &contour) {

	for(vector<Vec4i>::iterator it = defects.begin(); it != defects.end(); it++) {

		Vec4i& v = (*it);
		int startidx = v[0];
		Point ptStart(contour[startidx]);
		int endidx = v[1];
		Point ptEnd(contour[endidx]);
		int faridx = v[2];
		Point ptFar(contour[faridx]);
		float depth = v[3] / 256;

		line(image, ptStart, ptEnd, Scalar(255, 0, 0), 1);
		line(image, ptStart, ptFar, Scalar(255, 0, 0), 1);
		line(image, ptEnd, ptFar, Scalar(255, 0, 0), 1);
		circle(image, ptFar, 4, Scalar(255, 0, 0), 2);

		/*
		circle(image, ptEnd, 5, Scalar(255, 0, 0), 2);
		circle(image, ptStart, 5, Scalar(255, 0, 0), 2);
		circle( image, ptFar, 5, Scalar(255, 0, 0), 2);
		*/
	}
}

void ImageGeometricFeatureDetector::drawPoints(Mat &image, vector<Point> &points) {

	for (std::vector<Point>::const_iterator it = points.begin(); it != points.end(); it++) {

		circle(image, *it, 5, Scalar(255, 0, 0), 2);
	}
}

vector<Point> ImageGeometricFeatureDetector::convertToVectorOfPoints(const CvContourPolygon &contour) {

	vector<Point> result;
	vector<CvPoint> casted = static_cast<vector<CvPoint> >(contour);
	for (vector<CvPoint>::const_iterator it = casted.begin(); it != casted.end(); ++it) {

		result.push_back(Point((*it).x, (*it).y));

	}

	return result;
}
