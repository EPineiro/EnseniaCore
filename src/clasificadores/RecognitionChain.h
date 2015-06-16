/*
 * RecognitionChain.h
 *
 *  Created on: Jun 22, 2013
 *      Author: EPineiro
 */

#ifndef RECOGNITIONCHAIN_H_
#define RECOGNITIONCHAIN_H_

#include <opencv2/opencv.hpp>
#include <stdlib.h>

#include "RecognitionChainNode.h"

using namespace cv;

typedef std::pair<int, float> myPairType;

class RecognitionChain {

private:
	std::vector<RecognitionChainNode> recognizers;
	int notMatchingCode;

	std::map<int, float> possiblesSigns;
	std::map<int, float> errorsMargins;

	static bool compare(const myPairType &p1, const myPairType &p2) {
		return p1.second < p2.second;
	}

	int getMinErrorMarginSign() {

		myPairType min = *std::min_element(possiblesSigns.begin(), possiblesSigns.end(), &RecognitionChain::compare);
		return min.first;
	}

public:
	RecognitionChain() {

	}

	virtual ~RecognitionChain() {

	}

	inline void setNotMatchingCode(int notMatchingCode) {
		this->notMatchingCode = notMatchingCode;
	}

	void addNode(const RecognitionChainNode &node) {
		this->recognizers.push_back(node);
	}

	void loadErrorsMargins(const std::string &fileName) {

		FileStorage fs;
		fs.open(fileName, FileStorage::READ);

		FileNode node = fs["Signs"];

		for (FileNodeIterator it = node.begin(); it != node.end(); it++) {

			int code = (*it)["code"];
			float error = (*it)["error"];

			errorsMargins[code] = error;
		}
	}

	int recognize(const vector<float> &features) {

		for (std::vector<RecognitionChainNode>::iterator it =recognizers.begin(); it != recognizers.end(); it++) {

			if (it->recognize(features)) {

				//si reconocemos el signo, lo marcamos como un posible,
				//pero guardamos la diferencia entre el error de reconocimiento y el configurado aceptable.
				//luego chequeamos la diferencia con el margen de error para descartar falsos positivos
				possiblesSigns[it->getSignCode()] = abs(abs(it->getErrorMargin()) - abs(errorsMargins.find(it->getSignCode())->second));
			}
		}

		int signRecognized;
		//si no tenemos signos posibles, es porque no reconocimos nada
		if (possiblesSigns.empty()) {
			signRecognized = notMatchingCode;
		} else {
			signRecognized = getMinErrorMarginSign();
		}

		possiblesSigns.clear();

		return signRecognized;
	}
};

#endif /* RECOGNITIONCHAIN_H_ */
