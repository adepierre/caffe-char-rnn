#pragma once

#include <caffe/caffe.hpp>
#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <map>
#include <random>

class Trainer
{
public:
	/**
	* \brief Read the text file, create the vocabulary and initialize the trainer
	* \param solver Caffe solver (*.prototxt file)
	* \param snapshot Caffe snapshot file to resume training
	* \param textfile The text we want the classifier to learn
	* \param logfile The file name to log the loss during training
	* \param log_interval_ Loss is logged every log_interval_ iterations
	* \param sequence_length_ Length of one sequence of characters
	* \param batch_size_ Number of sequences in one training batch
	*/
	Trainer(const std::string &solver,
			const std::string &snapshot,
			const std::string &textFile, 
			const std::string &logFile_, 
			const int &log_interval_, 
			int& sequence_length_, 
			const int &batch_size_);
	~Trainer();

	/**
	* \brief Input new data in the net and perform one forward and one backward pass
	*/
	void Update();

	/**
	* \brief Set new data into the net
	* \param train If train is false, then set test data for evaluation
	*/
	void FeedNet(bool train);

	/**
	* \brief Clone the training net weights into the test net
	*/
	void CloneNet();

private:
	
	boost::shared_ptr<caffe::Solver<float> > solver;
	boost::shared_ptr<caffe::Net<float> > net;
	boost::shared_ptr<caffe::Net<float> > test_net;

	boost::shared_ptr<caffe::Blob<float> > blobData;
	boost::shared_ptr<caffe::Blob<float> > blobLabel;
	boost::shared_ptr<caffe::Blob<float> > blobClip;
	boost::shared_ptr<caffe::Blob<float> > blobLoss;


	boost::shared_ptr<caffe::Blob<float> > test_blobData;
	boost::shared_ptr<caffe::Blob<float> > test_blobLabel;
	boost::shared_ptr<caffe::Blob<float> > test_blobClip;
	boost::shared_ptr<caffe::Blob<float> > test_blobLoss;

	int sequence_length;
	int batch_size;
	bool random;

	std::ofstream logFile;
	int log_interval;
	
	std::map<char, int> charToInt;
	std::vector<char> intToChar;
	std::vector<float> rawDataIndexTrain;
	std::vector<float> rawDataIndexTest;

	std::vector<float> clip;
	std::vector<float> labels;
	std::vector<float> data;

	std::mt19937 randomEngine;
};

