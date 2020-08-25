# Information Exposure From Consumer IoT Devices

This site contains analysis code accompanying the paper "Information Exposure From Consumer IoT Devices: A Multidimensional, Network-Informed Measurement Approach" in proceedings of the ACM Internet Measurement Conference 2019 (IMC 2019), October 2019, Amsterdam, Netherlands. 

The official paper can be found at https://moniotrlab.ccis.neu.edu/imc19/. The site also contains instructions for requesting access to the full dataset.

The testbed code and documentation can be found at https://moniotrlab.ccis.neu.edu/tools/. Currently, it is deployed at both Northeastern University and Imperial College London. 

![GitHub Logo](lab.png)

Figure 1: The IoT Lab at Northeastern University.

## File Structure 

Each subdirectory shows samples for processing pcap files for destination, encryption, and content analysis.

- `destination/` - Code for Section 4 Destination Analysis - analyze the destinations that traffic is being sent to and received from.
- `encryption/` - Code for Section 5 Encryption Analysis - analyze whether traffic is encrypted or unencrypted.
- `model/` - Code for Section 6 Content Analysis - create machine learning models to predict the state of an IoT device using its network traffic.
- `moniotr/` - Code to automate experiments.
- `Getting_Started.md` - A step-by-step tutorial to get started analyzing data using each of the three analyses.
- `LICENSE.md` - The license for this software.
- `README.md` - This file. Contains an overview of the software.
- `lab.png` - A photo of the IoT Lab at Northeastern University.

## Datasets

We release the traffic (packet headers) from 34,586 controlled experiments and 112 hours of idle IoT traffic.

The naming convention for the data is `{country}{-vpn}/{device_name}/{activity_name}/{datetime}.{length}.pcap`. For example, `us/amcrest-cam-wired/power/2019-04-10_21:32:18.256s.pcap` is the traffic collected from device `amcrest-cam-wired` when `power` on at the time of `2019-04-10_21:32:18`, which lasts `256` seconds in the `us` lab without VPN.

To obtain access to the dataset, please follow the instructions on the paper webpage at https://moniotrlab.ccis.neu.edu/imc19. We require that you agree to the terms of our data sharing agreement. This is out of an abundance of caution to protect any private or security-sensitive information that we were unable to remove from the traces.

## Setup

Please see the [General Setup](Getting_Started.md#general-setup) section of Getting_Started.md for instructions on how to setup.

For more information about the pipelines and the contents of the code, see the READMEs for [destination analysis](destination/README.md), [encryption analysis](encryption/README.md), and [content analysis](model/README.md). Content analysis also has a page describing the machine learning models and contents of that directory in depth: [model/model_details.md](model/model_details.md).

For step-by-step instructions to get started analyzing data, see [Getting_Started.md](Getting_Started.md).

