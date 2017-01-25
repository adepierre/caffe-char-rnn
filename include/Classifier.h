#pragma once

#include <string>
#include <vector>
#include <memory>
#include <random>

#include <caffe/caffe.hpp>

class Classifier
{
public:
	
	/**
	* \brief Init net and solver, read the vocabulary file and create int_to_char vector
	* \param model_file Caffe prototxt file to generate the net
	* \param trained_file Caffe caffemodel file to load weights
	* \param vocabulary_file File created during training with all the characters
	* \param sequence_length_ Length of one sequence (in characters)
	* \param batch_size_ Number of sequences in one batch
	* \param temperature_ With high temperature the model does more mistakes but is less conservative
	* \param output_file_ File to write the characters in
	*/
	Classifier(const std::string &model_file,
			   const std::string &trained_file,
			   const std::string &vocabulary_file,
			   const int &sequence_length_,
			   const int &batch_size_,
			   const float &temperature_,
			   const std::string &output_file_);

	/**
	* \brief Predict N characters from an input sequence
	* \param sequence Input sequence, the model uses it to predict the next characters. If it's too short, random characters are added at the beginning, if it's too long, it's croped
	* \param N Size of the output characters
	* \param display If true the model writes the characters in the console as soon as it has predicted it
	* \return The vector of the N predicted characters
	*/
	std::vector<char> Predict(const std::vector<char> &sequence, int N, bool display);

private:

	/**
	* \brief Return the last prediction made by the net
	* \return The index in the vocabulary
	*/
	int GetLastPrediction();

	/**
	* \brief Return a random character from the vocabulary
	* \return A random valid character from the vocabulary
	*/
	char RandomChar();

private:

	std::shared_ptr<caffe::Net<float> > net;
	std::vector<char> int_to_char;
	std::vector<float> clip;


	boost::shared_ptr<caffe::Blob<float> > dataBlob;
	boost::shared_ptr<caffe::Blob<float> > clipBlob;
	boost::shared_ptr<caffe::Blob<float> > outputBlob;

	float temperature;
	std::mt19937 randomEngine;

	int sequence_length;
	int batch_size;

	std::string output_file_name;
	std::ofstream output_file;
};


