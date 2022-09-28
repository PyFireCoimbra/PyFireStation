# Build Instructions

### Requirements

- Python > = 3.7 (older versions may also work)
- pip
- virtualenv


## Windows 

TODO

## Linux/MacOS
 Tested on Big-Sur.
 
### Clone repository
```bash
git clone https://github.com/joao-aveiro/imfire-propagation.git
cd imfire-propagation
```

### Create Virtual Environment and install requirements
```bash
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

### Build Package
```bash
cd package
chmod +x build.sh
./build.sh
```

### Validate
```bash
cd dist
./PyFSation -h
```
