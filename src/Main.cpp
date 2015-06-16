#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <iterator>

#include "clasificadores/SVMClassifier.h"
#include "clasificadores/RecognitionChain.h"
#include "clasificadores/RecognitionChainNode.h"
#include "video/videoprocessor.h"
#include "video/SignRecognizerFrameProcessor.h"

using namespace std;
using namespace cv;

long featuresCount = 12;
int examplesCount = 119;
//int examplesCount = 30;

int featuresType = SignFeaturesDetector::USE_BLOBS;
int classifierType = Classifier::SVM_CLASSIFIER;

void testClassifier();
void testTrainedClassifierWithVideo();

void testRecognizeNumber();

int main(int argc, char** argv) {

	//testTrainedClassifierWithVideo();
	//testRecognizeNumber();

	waitKey(0);

	return 0;
}

void testTrainedClassifierWithVideo() {

/*	DIR *dp;
	struct dirent *dirp;

	VideoProcessor processor;

	processor.setInput(0);
	processor.displayOutput("Output Video");
	processor.setDelay(5);
	processor.setKeyToCaptureFrame(13);
	//processor.setOutput("resultados.avi", -1, 15.0);

	RecognitionChain chain;
	chain.setNotMatchingTextToDisplay("??");

	if ((dp = opendir("resources/svms_entrenados")) == NULL) {
		cout << "Error abriendo directorio con svm's entrenados" << endl;
	}

	while ((dirp = readdir(dp)) != NULL) {

		string fileName = string(dirp->d_name);

		if (fileName.find("clasificador") != string::npos) {

			string character = fileName.substr(fileName.find_first_of('_') + 1, 1);

			std::stringstream classifierFile;
			classifierFile << "resources/svms_entrenados/" << fileName;

			Classifier *classifier = Classifier::createClassifier(classifierType);
			classifier->loadClassifier(classifierFile.str());
			classifier->setFeaturesCount(featuresCount);
			RecognitionChainNode node(character, classifier);
			chain.addNode(node);

		}
	}

	closedir(dp);

	SignRecognizerFrameProcessor *frameProcesor = new SignRecognizerFrameProcessor();
	frameProcesor->setRoi(Rect (60, 170, 180, 220));
	frameProcesor->setFeaturesType(featuresType);
	frameProcesor->setRecognizer(chain);

	processor.setFrameProcessor(frameProcesor);

	processor.run();
*/
}
