# How to deploy ActiveTigger on a production linux based server?

The recommended method to deploy ActiveTigger on a linux based server is to use docker.

> [!INFO]
> If you wish to deploy without docker follow the docker configuration files as a guide to install and configure outside docker.

> [!WARNING]
> No test were made in a non-linux host server. Docker should work but the sections of this guide for GPU/HTTPS parts require a linux host.


## Requirements

To safely run activetigger for local use we recommend this configuration:

- 16 Go RAM
- 8 cores CPU
- 200 Go disk space
- 1 GPU with at least 8Go of VRAM (optional but recommended for large models)

If you want to scale for multiple projects :

- 4Go RAM by concurrent projects in memory
- 20Go disk space by concurrent projects

## Architecture

Activetigger is fueled by three services:

- a PostGreSQL database (using SQLite is possible but not recommended in production)
- a python API process
- a reverse proxy serving the built JavaScript client code and routing the API calls

## Prepare your server

### 1- docker

First you need to install docker on your host machine.
Please follow the [docker documentation](https://docs.docker.com/engine/install/).

Do not forget the post-install requirements. In particular to enable docker services

```bash
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

### 2- user system

It's a good practice to create a system user for the application to make sure maintenance can be achieved by multiple physical person using there own credentials to access the server.

```bash
sudo adduser activetigger
# add user to the docker group
sudo adduser activetigger docker
```

You might want to deny SSH access by this generic user:

```bash
sudo vi /etc/ssh/ssh_config
...
DenyUsers activetigger
...
```

Restart SSH

```bash
sudo systemctl restart ssh
```

### 3- create the app directory

Depending on your server configuration carefully chose where to store the app directory.
This is where all the data will be stored so make sure to have enough disk space.

```bash
sudo mkdir /opt/activetigger
sudo chown activetigger:activetigger /opt/activetigger
```

### 4- clone source code

Use the generic system user to clone the source code in the app directory.
If your system does not already have git installed install it first.
Make sure to use the production branch.

```bash
sudo su activetigger
cd /opt/activetigger
git clone https://github.com/activetigger/activetigger.git .
git checkout production
```

### 5- Configure your environment, especially data path

The environment variables are defined in `docker/.env` : passwords (root and database), server ports and load, secrets, etc.

An important parameter is the data path.

If you need the application data (projects data, models data and database data) to be mounted in a specific point in your filesystem you can use the `DATA_PATH` env variable.

By default, the data will be stored in `activetigger/docker/api` where the filesystem is already created.

But if you change the `DATA_PATH` env variable you need to create the right folder architecture in this directory with the right user privileges **before** staring docker.

Here is a small script you can use the create the `DATA_PATH` directory.

```bash
# make sure to use the application user
sudo su activetigger
# first edit the DATA_PATH variable in .env
vi docker/.env
...
# for instance let's use /data
DATA_PATH=/home/activetigger/data
...
source .env
mkdir -p $DATA_PATH
mkdir -p $DATA_PATH/venv
mkdir -p $DATA_PATH/models
mkdir -p $DATA_PATH/projects/static
```

## NVIDIA GPU

### NVIDIA driver

If the command `nvidia-smi` does not work you need to install the NVIDIA driver on your host machine.

You have to install the Nvidia card driver available here: https://www.nvidia.com/en-us/drivers/
It is recommended to use the package method rather than the "runner" one.

For Ubuntu for instance we've successfully tested the package method documented here: https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

You can also follow the guidelines here : https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation-network

### NVIDIA Container Toolkit

To use the GPU inside docker you need to install the NVIDIA Container Toolkit.

Make sure to install NVIDIA Container Toolkit by following [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

As an example in an ubuntu here is the process:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test your config

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
Tue Jul  1 10:55:16 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.169                Driver Version: 570.169        CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro RTX 5000                Off |   00000000:00:06.0 Off |                    0 |
|  0%   39C    P2             25W /  230W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## configure docker

Edit the `.env` file to suite your needs and more particularly.

If you use activetigger locally, you can use the dev configuration. If you need to deploy it on a production server, you can use the prod configuration.

```bash
MODE=prod
# Required domain name to create SSL certificate on. Without that we can't do HTTPS which is unsecure.
DOMAIN=yourdomain.com
# email which will used to create the SSL certificate
CERTBOT_EMAIL=admin@yourdomain.com
# your private secret seed
SECRET="a random 32 url-safe base64-encoded"
```

## start

If you server has a configured nvidia GPU:

```bash
docker compose -p activetigger -f docker-compose.yml -f docker-compose.nvidia.yml -f docker-compose.prod.yml up -d
```

If not:

```bash
docker compose -p activetigger -f docker-compose.yml -f docker-compose.prod.yml up -d
```
