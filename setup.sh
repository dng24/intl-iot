#!/bin/bash

#setup python3.6 environment
sudo apt-get -y update
sudo apt-get -y install virtualenv libpcap-dev libpq-dev python3-dev python3.6-tk g++ tshark
virtualenv -p python3.6 env
source env/bin/activate

#for all pipelines
pip install numpy scipy pyshark geoip2 matplotlib dpkt pycrypto IPy pcapy scapy Impacket \
    mysql-connector-python-rf pandas tldextract python-whois ipwhois psutil seaborn

#for model pipeline
pip install 'statsmodels==0.12.0' 'certifi==2020.6.20' 'joblib==0.16.0' 'numpy==1.19.1' 'pandas==1.1.0' \
    'python-dateutil==2.8.1' 'pytz==2020.1' 'scikit-learn==0.23.2' 'scipy==1.5.2' \
    'six==1.15.0' 'threadpoolctl==2.1.0' 'seaborn==0.10.1' 'whois==0.9.7'

deactivate
