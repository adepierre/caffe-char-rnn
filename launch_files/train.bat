"../bin/Release/caffe_char_rnn" ^
--train=true ^
--gpu=true ^
--sequence_length=75 ^
--batch_size=25 ^
--solver=caffe_char_rnn_solver.prototxt ^
--snapshot= ^
--logfile=log.txt ^
--log_interval=100 ^
--textfile=shakespeare.txt