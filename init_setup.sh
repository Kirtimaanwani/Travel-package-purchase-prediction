echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8" # change py version as per your need
conda create --prefix ./venv python=3.8 -y
echo [$(date)]: "activate env"
source activate ./venv
echo [$(date)]: "intalling dev requirements"
pip install -r requirements.txt
echo [$(date)]: "END"
