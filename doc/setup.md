### Installation
**1. Download pyenv and EMAN2**  
```bash
git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv
wget https://cryoem.bcm.edu/cryoem/static/software/release-2.3/eman2.3.linux64.sh
```

**2. Setup environment for pyenv and EMAN2**  
* Add the following lines to `$HOME/.bash_profile`
```bash
# Setting for EMAN2.3
export PATH="$PATH:$HOME/EMAN2/bin"

# Setting for pyenv and pyenv-virtualenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
fi
```

**3. Update your environment (pyenv will be installed)**
```bash
source $HOME/.bash_profile
```

**4. Install pyenv-virtualenv**
```bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.bash_profile
source $HOME/.bash_profile
```

**5. Install EMAN2**
* Run the following command
```bash
bash eman2.3.linux64.sh -b
```

### Create DEFMap implementation environment in pyenv
* Install miniconda3.
```bash
pyenv install miniconda3-4.2.12
```

* Create virtual environment using conda
```bash
conda create -n home python=3.6
pyenv global miniconda3-4.2.12/envs/home
conda create -n defmap python=3.6
pyenv shell miniconda3-4.2.12/envs/defmap
```

* Install the dependency libraries.
```
conda install -c acellera -c psi4 htmd=1.15.2
conda install keras-gpu=2.2.4 tensorflow-gpu=1.13.1 cudatoolkit=10.0
# if the version of CUDA Toolkit installed on your server is 9.0, please specify cudatoolkit=9.0
```
If you encounter errors (e.g. "PackageNotFound"), please try below commands.
```
conda install -c conda-forge -c acellera -c psi4 htmd=1.15.2
conda install -c conda-forge keras-gpu=2.2.4 tensorflow-gpu=1.13.1 cudatoolkit=10.0
```
If the above instruction does not work, please create an issue as "Installation issue".
