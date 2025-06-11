# A Digital Twin Framework for Urban Parking Management and Mobility Forecasting

## Abstract
Rapid urbanization and population growth have created significant challenges in urban mobility management, such as traffic congestion, inefficient public transportation, and environmental pollution. This paper presents A framework for urban parking management and mobility forecasting, development and implementation of a Digital Twin (DT) aimed at addressing issues within the context of smart mobility. The framework integrates a wide range of historical and real-time data, including parking meter transactions, revenue records, street occupancy rates, parking violations, and sensor-based parking slot utilization. Additionally, the data encompass weather conditions, temporal patterns (such as weekdays and peak hours), and agent shift schedules, offering a comprehensive dataset for analyzing and optimizing urban mobility dynamics. Descriptive statistics are used to identify key patterns, while advanced Machine Learning (ML) and Deep Learning (DL) algorithms enhance predictive and generative analytics, forecasting parking demand and simulating various mobility scenarios. These insights, combined with visualization tools, map data onto the urban landscape, enabling spatial planning and resource allocation. Moreover, the integration of Generative Artificial Intelligence (GenAI) models significantly improves the system's capabilities, generating realistic ``what-if'' scenarios that allow for virtual testing of mobility strategies before real-world implementation.
The results highlight the framework potential to improve urban mobility management, especially improving parking meter placement and enhancing the quality of urban mobility for users by reducing inefficiencies and improving accessibility. Tested on real-world data from the city of Caserta, the proposed framework has proven robust and adaptable, although expanding the dataset and refining specific components are necessary for fully realizing its potential and ensuring sustainable urban planning.


![Alt text](DT.png)




## Acknowledgments
The authors would like to thank K-city srl company (https://www.k-city.eu/) for their support, collaboration, and provision of the data. In particular, special thanks go to Dr. Giuseppe Morelli and Dr. Sebastiano Spina. The authors also thank the city of Caserta, including its mayor and staff, for their support.



## Data Availability
The data used in this study are not publicly available due to confidentiality agreements and data protection policies. However, all relevant specifications regarding the data structure, format, and preprocessing procedures are provided in the accompanying GitHub repository. 
To facilitate adaptation of the framework to new settings, we also provide sample files containing synthetic or anonymized single entries for each dataset. These examples demonstrate the required input format and schema for integration with the predictive, generative, and scheduling modules. For the agent shift calendar, a compiled and transformed version is included to illustrate how the anonymized data feeds into the optimization model.
Researchers interested in accessing the data may contact the corresponding author to discuss potential access arrangements under appropriate confidentiality agreements.
The weather and air quality data, which are open source, can be downloaded from \url{https://open-meteo.com}, while the Points of Interest data, obtained through OpenStreetMap, can be accessed at \url{https://www.openstreetmap.org}.


## Installation

### 0. Requirements
All the code can be executed through Docker with nvidia-container-toolkit set up.

### 1. Build of the Docker Image
Run the following command from the terminal:

```sh
docker compose build
```

this will install all the necessary for the execution of each step.

### 2. Running the newly generated image as container

Run the following command from the terminal:

```sh
docker compose run --rm preprocessing
```

this will start a new shell session in the generated container

## Execution
### 1. Forecasting

From the shell, run the following for the forecasting:

```sh
bash run_forecasting.sh
```

### 2. Generation
For the generation, run the following:

```sh
bash run_generation.sh
```

### 3. Scheduling
For the agent scheduling for 12th January 2025, run the following:

```sh
bash run_scheduling.sh 2025-01-12
```