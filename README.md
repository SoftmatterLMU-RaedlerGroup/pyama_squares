# Create square ROIs for PyAMA

## Installation
Clone or download this repository, open a command prompt inside it and run:

```
mkdir env
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Windows or Anaconda users may need to adapt the commands.

When using Anaconda in Windows, the script `pyama_squares.cmd` may be used to open a command window with the alias `pyama_squares` defined.
In this case, you may need to adjust the script to match your Anaconda installation path.

## Usage
### Input
`*.pickle` file exported using PyAMA’s “Pickle maximum bounding box” command

### Output
numpy file suitable for import in PyAMA as segmentation/binary stack

### Details on usage
See `python pyama_squares.py -h` for more information.
