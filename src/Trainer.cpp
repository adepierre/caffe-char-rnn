#include "Trainer.h"
#include <chrono>


Trainer::Trainer(const std::string &solver_, const std::string &snapshot,
	const std::string &textFile, const std::string &logFile_,
	const int &log_interval_, int& sequence_length_,
	const int &batch_size_) :
	sequence_length(sequence_length_), batch_size(batch_size_),
	log_interval(log_interval_)
{
	//Read text file
	std::ifstream file(textFile, std::ios::in);
	if (!file)
	{
		std::cerr << "Error, text file not found" << std::endl;
		return;
	}
	else
	{
		std::cout << "Reading " + textFile << std::endl;

		file.seekg(0, std::ios::end);
		size_t fileSize = file.tellg();
		file.clear();
		file.seekg(0, std::ios::beg);

		int currentCharIndex = 0;
		int fileCounter = 0;
		char currentChar;
		while ((currentChar = file.get()) != EOF)
		{
			auto iterator = charToInt.find(currentChar);
			//If the character as already been seen, just add the int value of the character
			if (iterator != charToInt.end())
			{
				if (fileCounter < 0.9*fileSize)
				{
					rawDataIndexTrain.push_back((float)iterator->second);
				}
				else
				{
					rawDataIndexTest.push_back((float)iterator->second);
				}
			}
			//Else, create a new int value and add it
			else
			{
				charToInt[currentChar] = currentCharIndex;
				intToChar.push_back(currentChar);
				if (fileCounter < 0.8*fileSize)
				{
					rawDataIndexTrain.push_back((float)currentCharIndex);
				}
				else
				{
					rawDataIndexTest.push_back((float)currentCharIndex);
				}
				currentCharIndex++;
			}
			fileCounter++;
		}
		file.close();
		std::ofstream vocabFile("vocabulary_" + std::to_string(intToChar.size()) + ".txt", std::ios::out | std::ios::trunc);
		for (int i = 0; i < intToChar.size(); ++i)
		{
			vocabFile << intToChar[i];
		}
		vocabFile.close();
	}

	//Create caffe objects (solver + net)
	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(solver_, &solver_param);
	
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	net = solver->net();

	if (snapshot.empty())
	{
		std::cout << "Starting new training" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot << std::endl;
		solver->Restore(snapshot.c_str());
	}

	//Get input and output blobs
	blobData = net->blob_by_name("data");
	blobClip = net->blob_by_name("clip");
	blobLabel = net->blob_by_name("label");
	blobLoss = net->blob_by_name("loss");

	//Initialize input data
	clip = std::vector<float>(sequence_length * batch_size);
	for (int i = 0; i < clip.size(); ++i)
	{
		clip[i] = (i < batch_size) ? 0.0f : 1.0f;
	}
	labels = std::vector<float>(sequence_length*batch_size);
	data = std::vector<float>(sequence_length*batch_size);

	blobClip->set_cpu_data(clip.data());

	//Initialize random engine
	randomEngine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	//Initialize log file
	if (!logFile_.empty())
	{
		logFile.open(logFile_, std::ios::out | std::ios::app);
	}
	if (logFile.is_open())
	{
		logFile << "Iteration;Training loss;Iteration;Validation loss" << "\n";
	}
}


Trainer::~Trainer()
{
	solver->Snapshot();
}

void Trainer::Update()
{
	//If it's test time
	if (solver->iter() % solver->param().display() == 0 && solver->iter() > 0)
	{
		CloneNet();
		float meanLoss = 0.0f;
		for (int i = 0; i < 20; ++i)
		{
			FeedNet(false);
			test_net->Forward();
			meanLoss += test_blobLoss->cpu_data()[0];
		}
		meanLoss /= 20.0f;
		std::cout << "Validation loss : " << meanLoss << std::endl;
		if (logFile.is_open())
		{
			logFile << ";;" << solver->iter() << ";" << meanLoss << "\n";
		}
	}

	//Forward + backward pass
	FeedNet(true);
	solver->Step(1);

	if (logFile.is_open() && solver->iter() % log_interval == 0)
	{
		logFile << solver->iter() << ";" << blobLoss->cpu_data()[0] << "\n";
	}
}

void Trainer::FeedNet(bool train)
{
	//Select the right objects (train or test)
	std::vector<float> *rawDataIndex = train ? &rawDataIndexTrain : &rawDataIndexTest;
	boost::shared_ptr<caffe::Blob<float> > blobData_ = train ? blobData : test_blobData;
	boost::shared_ptr<caffe::Blob<float> > blobLabel_ = train ? blobLabel : test_blobLabel;

	//Create input data, the data must be in the order
	//seq1_char1, seq2_char1, ..., seqBatch_Size_char1, seq1_char2, ... , seqBatch_Size_charSequence_Length
	//As seq1_charSequence_Length == seq2_charSequence_Lenght-1 == seq3_charSequence_Length-2 == ...  we can perform block copy for efficiency
	//Labels are the same with an offset of +1

	//If batch_size == 1 we don't need to copy, raw data have the right order
	if (batch_size == 1)
	{
		int currentCharacter = std::uniform_int_distribution<int>(0, rawDataIndex->size() - sequence_length - 1)(randomEngine);

		//Feed the net with input data and labels (clips are always the same)
		blobData_->set_cpu_data(rawDataIndex->data() + currentCharacter);
		blobLabel_->set_cpu_data(rawDataIndex->data() + currentCharacter + 1);
	}
	//If batch_size > 1, we have to re-order the data according to caffe input specification for LSTM layer
	else
	{
		for (int i = 0; i < batch_size; ++i)
		{
			int currentCharacter = std::uniform_int_distribution<int>(0, rawDataIndex->size() - sequence_length - 2)(randomEngine);
			for (int j = 0; j < sequence_length; ++j)
			{
				data[batch_size*j + i] = rawDataIndex->data()[currentCharacter + j];
				labels[batch_size*j + i] = rawDataIndex->data()[currentCharacter + j + 1];
			}
		}

		//Feed the net with input data and labels (clips are always the same)
		blobData_->set_cpu_data(data.data());
		blobLabel_->set_cpu_data(labels.data());
	}
}

void Trainer::CloneNet()
{

	caffe::NetParameter net_param;
	net->ToProto(&net_param);
	net_param.mutable_state()->set_phase(caffe::Phase::TEST);
	if (test_net == nullptr)
	{
		test_net.reset(new caffe::Net<float>(net_param));

		test_blobData = test_net->blob_by_name("data");
		test_blobClip = test_net->blob_by_name("clip");
		test_blobLabel = test_net->blob_by_name("label");
		test_blobLoss = test_net->blob_by_name("loss");

		test_blobClip->set_cpu_data(clip.data());
	}
	else
	{
		test_net->CopyTrainedLayersFrom(net_param);
	}
}