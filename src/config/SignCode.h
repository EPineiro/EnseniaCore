/*
 * SignCode.h
 *
 *  Created on: Sep 12, 2013
 *      Author: EPineiro
 */

#ifndef SIGNCODE_H_
#define SIGNCODE_H_

#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

class SignCode {

public:
	int code;
	string name;

	SignCode() : code(0), name() {}

	SignCode(const int _code, const string _name) : code(_code), name(_name) {}

	SignCode(const SignCode& obj) : code(obj.code), name(obj.name) {}

	void read(const FileNode& node) {

		code = (int) node["code"];
		name = (string) node["name"];

		if(name == "") {
			stringstream ss;
			ss << (int)node["name"];
			name = ss.str();
		}
	}

	void write(FileStorage &fs) const {

		fs << "{" << "code" << code << "name" << name <<"}";
	}
};

static void write(FileStorage &fs, const std::string &name, const SignCode &sign) {

	sign.write(fs);
}

static void read(const FileNode &node, SignCode &sign, const SignCode &default_value = SignCode()) {

	if(node.empty())
		sign = default_value;
	else
		sign.read(node);
}

static ostream& operator <<(ostream &out, const SignCode sign) {

	out << "Instance " << "{";
	out << "code: " << sign.code << " name: " << sign.name;
	out << "}";

	return out;
}

#endif /* SIGNCODE_H_ */
