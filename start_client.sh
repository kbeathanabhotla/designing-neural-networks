export PYTHON_PATH=$PYTHON_PATH:./client
python ./client/com/designingnn/client/start_server.py -server_host 127.0.0.1 -server_port 8080 -client_port 8081 -metadata_dir "/home/sai/mtss_proj/metadata" -data_dir "/home/sai/mtss_proj/data/mnist"
