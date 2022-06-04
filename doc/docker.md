## Dependency
- nvidia-docker: >20.x

## Build

1. Build docker image from Dockerfile

```bash
docker build -t defmap .
```

2. Run a container in interactive mode

```bash
docker run -it --gpus 0 defmap bash
```


## How to switch EMAN2 and DEFMap environments

If you want to use EMAN2 command, run the following command:

```bash
eval "$(/root/eman2-sphire-sparx/bin/conda shell.bash hook)"
```

If you want to use DEFMap, run the follwoing command:

```bash
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda actiavte defmap
```
