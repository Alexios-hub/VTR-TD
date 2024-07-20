FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# install openssh and openssl 
RUN set -ex \
	&& apt-get -o Acquire::Check-Valid-Until=false  -o Acquire::Check-Date=false   update \
	&& apt-get install -y --no-install-recommends openssh-client openssh-server \
	&& mkdir -p /var/run/sshd \
	&& /usr/bin/ssh-keygen -A

# Allow OpenSSH to talk to containers without asking for confirmation, and import default key
RUN set -ex \
	&& cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
	&& echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
	&& cat /etc/ssh/sshd_config | grep -v  PermitRootLogin> /etc/ssh/sshd_config.new \
	&& echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.new \
	&& mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
	&& mv /etc/ssh/sshd_config.new /etc/ssh/sshd_config \
	&& mkdir -p /root/.ssh \
	&& echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDp4ajLJi3urg83XZ+XL3C1ybdmpjYl31AbJ6hnFZKWr8wtYjuvBophXKcmdVoU/OFL0BYr1DLEMdlI2wNBQ/2jAPa9sQEi55UcXvKx9Fs7VHGl35I0e+fM0jHZDOB+dpl+Kk3gectU2P6Lcb9H1aaJlA9O2sG9p3fCXOg3qRHY7t10rHqOxsUJi3ufGU+T0xuYWzWFDX6n1PL2ZzerRsU293hMfRdbUCJy+YeT9E7V/k00201WOxYzCSQY9V708FfIcTumDDIfrD+Q2SfG5tgSqw8+L1hFTe2vq4h63kH8UXETy5Ae/EUhdwNyG/ph3WgFaspcevxRItHwDeXaXyKq9f9X3dmyqTvmtRr+FEpalZIf7/Ny9DBLfcLExjzdFBjLl0+u6RCMlD0xvl02fAJQTDl0JYonA/A12PDexRKDTVTGefMW0JzZf0FWysuGpADnVoGNlbxh10h87u3cFDI5jeTka4lyqmZKxNtvkvt6uL61RDX+B98lgP3fnryNKtU= root@node-1" >> /root/.ssh/authorized_keys \
	&& chmod 700 /root/.ssh \
	&& chmod 600 /root/.ssh/authorized_keys

# Install essential Ubuntu packages
RUN apt-get update && apt-get install -y \
    rsync \
    build-essential \
    git \
    wget \
    vim \
    curl \
    zip \
    iputils-ping \
    zlib1g-dev \
    unzip \
    tmux \
    pkg-config \
    libgl-dev \
    libblas-dev \
    liblapack-dev \
    graphviz \
    libhdf5-dev \
    swig \
    apt-transport-https \
    lsb-release \
    libpng-dev \
    ca-certificates &&\
    apt-get clean &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# init conda
RUN /opt/conda/condabin/conda init
RUN conda config --append channels conda-forge
      
# Install essential Python packages
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    blackcellmagic \
    pytest \
    pytest-cov \
    numpy \
    matplotlib \
    scipy \
    pandas \
    jupyter \
    scikit-learn \
    scikit-image \
    seaborn \
    graphviz \
    gpustat \
    h5py \
    gitpython \
    ptvsd \
    Pillow \
    opencv-python \
    prefetch_generator \
    optuna \
    einops \
    PyYAML \
    tqdm \
    tensorboard \
    jupyterlab \
    absl-py==2.1.0 \
    anyio==4.4.0 \
    argon2-cffi==23.1.0 \
    argon2-cffi-bindings==21.2.0 \
    arrow==1.3.0 \
    asttokens==2.4.1 \
    async-lru==2.0.4 \
    attrs==23.2.0 \
    babel==2.15.0 \
    beautifulsoup4==4.12.3 \
    bleach==6.1.0 \
    braceexpand==0.1.7 \
    certifi==2024.7.4 \
    cffi==1.16.0 \
    charset-normalizer==3.3.2 \
    comm==0.2.2 \
    debugpy==1.8.2 \
    decorator==5.1.1 \
    decord==0.6.0 \
    defusedxml==0.7.1 \
    einops==0.8.0 \
    einops-exts==0.0.4 \
    exceptiongroup==1.2.1 \
    executing==2.0.1 \
    fastjsonschema==2.20.0 \
    filelock==3.15.4 \
    fqdn==1.5.1 \
    fsspec==2024.6.1 \
    ftfy==6.2.0 \
    grpcio==1.64.1 \
    h11==0.14.0 \
    httpcore==1.0.5 \
    httpx==0.27.0 \
    huggingface-hub==0.23.4 \
    idna==3.7 \
    ipykernel==6.29.5 \
    ipython==8.26.0 \
    ipywidgets==8.1.3 \
    isoduration==20.11.0 \
    jedi==0.19.1 \
    jinja2==3.1.4 \
    json5==0.9.25 \
    jsonpointer==3.0.0 \
    jsonschema==4.22.0 \
    jsonschema-specifications==2023.12.1 \
    jupyter==1.0.0 \
    jupyter-client==8.6.2 \
    jupyter-console==6.6.3 \
    jupyter-core==5.7.2 \
    jupyter-events==0.10.0 \
    jupyter-lsp==2.2.5 \
    jupyter-server==2.14.1 \
    jupyter-server-terminals==0.5.3 \
    jupyterlab==4.2.3 \
    jupyterlab-pygments==0.3.0 \
    jupyterlab-server==2.27.2 \
    jupyterlab-widgets==3.0.11 \
    markdown==3.6 \
    markupsafe==2.1.5 \
    matplotlib-inline==0.1.7 \
    mistune==3.0.2 \
    mpmath==1.3.0 \
    nbclient==0.10.0 \
    nbconvert==7.16.4 \
    nbformat==5.10.4 \
    nest-asyncio==1.6.0 \
    networkx==3.3 \
    notebook==7.2.1 \
    notebook-shim==0.2.4 \
    numpy==1.26.4 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nccl-cu12==2.20.5 \
    nvidia-nvjitlink-cu12==12.5.82 \
    nvidia-nvtx-cu12==12.1.105 \
    overrides==7.7.0 \
    packaging==24.1 \
    pandas==2.2.2 \
    pandocfilters==1.5.1 \
    parso==0.8.4 \
    pexpect==4.9.0 \
    pillow==10.4.0 \
    platformdirs==4.2.2 \
    prometheus-client==0.20.0 \
    prompt-toolkit==3.0.47 \
    protobuf==4.25.3 \
    psutil==6.0.0 \
    ptyprocess==0.7.0 \
    pure-eval==0.2.2 \
    pycparser==2.22 \
    pygments==2.18.0 \
    python-dateutil==2.9.0.post0 \
    python-json-logger==2.0.7 \
    pytz==2024.1 \
    pyyaml==6.0.1 \
    pyzmq==26.0.3 \
    qtconsole==5.5.2 \
    qtpy==2.4.1 \
    referencing==0.35.1 \
    regex==2024.5.15 \
    requests==2.32.3 \
    rfc3339-validator==0.1.4 \
    rfc3986-validator==0.1.1 \
    rpds-py==0.18.1 \
    safetensors==0.4.3 \
    send2trash==1.8.3 \
    sentencepiece==0.2.0 \
    six==1.16.0 \
    sniffio==1.3.1 \
    soupsieve==2.5 \
    stack-data==0.6.3 \
    sympy==1.12.1 \
    tensorboard==2.17.0 \
    tensorboard-data-server==0.7.2 \
    terminado==0.18.1 \
    timm \
    tinycss2==1.3.0 \
    tokenizers==0.19.1 \
    tomli==2.0.1 \
    torch==2.3.1 \
    torchvision==0.18.1 \
    tornado==6.4.1 \
    tqdm==4.66.4 \
    traitlets==5.14.3 \
    transformers==4.42.3 \
    triton==2.3.1 \
    types-python-dateutil==2.9.0.20240316 \
    typing-extensions==4.12.2 \
    tzdata==2024.1 \
    uri-template==1.3.0 \
    urllib3==2.2.2 \
    wcwidth==0.2.13 \
    webcolors==24.6.0 \
    webdataset==0.2.86 \
    webencodings==0.5.1 \
    websocket-client==1.8.0 \
    werkzeug==3.0.3 \
    widgetsnbextension==4.0.11

## configure jupyterlab                                                                                                                                            
RUN set -ex \
	&& mkdir /etc/jupyter/ \
	&& wget -P /etc/jupyter/  https://raw.githubusercontent.com/Egolas/jupyter_gpu/master/jupyter_notebook_config.py \
	&& wget -P /etc/jupyter/ https://raw.githubusercontent.com/Egolas/jupyter_gpu/master/custom.js 

# Export port for Jupyter Notebook
EXPOSE 8888

# By default start running jupyter notebook
ENTRYPOINT ["jupyter", "lab", "--allow-root"]

