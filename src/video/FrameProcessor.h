/*
 * FrameProcessor.h
 *
 *  Created on: May 6, 2013
 *      Author: EPineiro
 */

#ifndef FRAMEPROCESSOR_H_
#define FRAMEPROCESSOR_H_

#include <opencv2/core/core.hpp>

// The frame processor interface
class FrameProcessor {

  public:
	// processing method
	virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;
};

#endif /* FRAMEPROCESSOR_H_ */
