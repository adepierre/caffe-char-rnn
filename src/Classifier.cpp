#include "Classifier.h"

#include <chrono>
#include <algorithm>
#include <numeric>

/**
* \brief Returns the index of the top value of a vector v
* \param v The vector
* \return The index of the top value
*/
static int Argmax(const std::vector<float>& v)
{
	if (v.size() == 0)
	{
		return -1;
	}

	int index = 0;
	float maxi = v[0];
	for (size_t i = 0; i < v.size(); ++i)
	{
		if (v[i] > maxi)
		{
			index = i;
			maxi = v[i];
		}
	}

	return index;
}

Classifier::Classifier(const std::string &model_file,
					   const std::string &trained_file,
					   const std::string &vocabulary_file,
					   const int &sequence_length_,
					   const int &batch_size_,
					   const float &temperature_,
					   const std::string &output_file_):
	sequence_length(sequence_length_),
	batch_size(batch_size_),
	output_file_name(output_file_)
{
	/* Load the network. */
	net.reset(new caffe::Net<float>(model_file, caffe::TEST));
	if (!trained_file.empty())
	{
		net->CopyTrainedLayersFrom(trained_file);
	}

	dataBlob = net->blob_by_name("data");
	clipBlob = net->blob_by_name("clip");
	outputBlob = net->blob_by_name("ip1");

	/* Load labels. */
	std::ifstream labelFile(vocabulary_file.c_str());
	CHECK(labelFile) << "Unable to open labels file " << vocabulary_file;
	
	char currentChar;
	while ((currentChar = labelFile.get()) != EOF)
	{
		int_to_char.push_back(currentChar);
	}

	//Create clip data
	clip = std::vector<float>(sequence_length*batch_size);
	for (int i = 1; i < clip.size(); ++i) 
	{
		clip[i] = (i < batch_size) ? 0.0f : 1.0f;
	}

	clipBlob->set_cpu_data(clip.data());

	//Temperature can't be negative
	temperature = abs(temperature_);
	if (temperature > 1.0f)
	{
		temperature = 1.0f;
	}

	randomEngine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

std::vector<char> Classifier::Predict(const std::vector<char> &sequence, int N, bool display)
{
	//Crop sequence if it has too many characters, or add random characters at the beginning if it doesn't have enough
	std::vector<char> correctLengthSequence(sequence_length);
	if (sequence.size() == sequence_length)
	{
		correctLengthSequence = sequence;
	}
	else if (sequence.size() > sequence_length)
	{
		correctLengthSequence = std::vector<char>(sequence.end() - sequence_length, sequence.end());
	}
	else
	{
		for (int i = 0; i < sequence_length - sequence.size(); ++i)
		{
			correctLengthSequence[i] = RandomChar();
		}
		std::copy(sequence.begin(), sequence.end(), correctLengthSequence.end() - sequence.size());
	}
	//The N returned characters
	std::vector<char> predictions(N);

	//The deque representing the current sequence
	std::deque<int> input;

	for (int i = 0; i < correctLengthSequence.size(); ++i)
	{
		for (int j = 0; j < int_to_char.size(); ++j)
		{
			if (correctLengthSequence[i] == int_to_char[j])
			{
				input.push_back(j);
				break;
			}


			if (j == int_to_char.size())
			{
				std::cerr << "Error, the input sequence contains at least one unknown character: " << sequence[i] << std::endl;
				return std::vector<char>();
			}
		}
	}

	if (!output_file_name.empty())
	{
		output_file.open(output_file_name, std::ios::out | std::ios::app);
	}


	for (int i = 0; i < N; ++i)
	{
		//The batch is filled with 0 except for the first sequence which is the one we want to use for prediction
		std::vector<float> inputVector(sequence_length*batch_size, 0.0f);
		for (int j = 0; j < sequence_length; ++j)
		{
			inputVector[j*batch_size] = (float)input[j];
		}

		dataBlob->set_cpu_data(inputVector.data());
		net->Forward();

		int prediction = GetLastPrediction();


		//Add the new prediction and discard the oldest one
		input.push_back(prediction);
		input.pop_front();

		predictions[i] = int_to_char[prediction];

		if (display)
		{
			std::cout << int_to_char[prediction];
		}

		if (!output_file_name.empty())
		{
			output_file << int_to_char[prediction];
		}
	}

	if (!output_file_name.empty())
	{
		output_file << std::endl << std::endl << std::endl << std::endl;
		output_file.close();
	}

	return predictions;
}

int Classifier::GetLastPrediction()
{
	const float* data = outputBlob->cpu_data();

	//Get the probabilities for the last character of the first sequence in the batch
	int offset = (sequence_length - 1)*batch_size*int_to_char.size();
	std::vector<float> vectorData(data + offset, data + offset + int_to_char.size());

	//If no temperature, return directly the character with the best score
	if (temperature == 0.0f)
	{
		return Argmax(vectorData);
	}
	//Else, compute the probabilities with the temperature and select a character according to them
	else
	{
		std::vector<float> proba(vectorData.size()), accumulatedProba(vectorData.size());

		auto maxValue = std::max_element(vectorData.begin(), vectorData.end());
		for (int i = 0; i < vectorData.size(); ++i)
		{
			//The max value is substracted for numerical stability
			proba[i] = exp((vectorData[i] - *maxValue) / temperature);
		}

		float expoSum = std::accumulate(proba.begin(), proba.end(), 0.0f);

		proba[0] /= expoSum;
		accumulatedProba[0] = proba[0];
		float randomNumber = std::uniform_real_distribution<float>(0.0f, 1.0f)(randomEngine);

		for (int i = 1; i < proba.size(); ++i)
		{
			//Return the first index for which the accumulated probability is bigger than the random number
			if (accumulatedProba[i - 1] > randomNumber)
			{
				return i-1;
			}
			proba[i] /= expoSum;
			accumulatedProba[i] = accumulatedProba[i - 1] + proba[i];
		}

		//If we are here, it's the last character
		return proba.size() - 1;
	}
}

char Classifier::RandomChar()
{
	return int_to_char[std::uniform_int_distribution<int>(0, int_to_char.size() - 1)(randomEngine)];
}