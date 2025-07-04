In order to download the package, you can simply run the following command from the terminal: 

git clone https://github.com/aarushehe/w-oStreamlit

In order to install all required Python modules, simply run:

To create a virtual environment so that all the downloaded dependencies are isolated then :

py -3.10 -m venv tf-env

To activate the environment write:

.\tf-env\Scripts\activate

Then write the below commands for the installation of the necessary dependencies-

pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

pip install tf-keras

webrtcvad_wheels-2.0.14-cp310-cp310-win_amd64

pip install -r requirements.txt

To run the camera in terminal, write:

python camera_app.py

To run this project on streamlit, write:

streamlit run streamlit.py


to start mongoDB manually - 

in the terminal or the command prompt write the following commands -

cd "path to the monogdb bin"

then,

>mongod.exe --dbpath ..\data\db
