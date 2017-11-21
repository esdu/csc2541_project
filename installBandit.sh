# This file collects the bandit code and puts it here

pushd ~
rm -rf RecommendationSystems # Remove any existing stuff
git clone https://github.com/scheeloong/RecommendationSystems
cd RecommendationSystems

# Installation instruction for others
sudo pip install -r requirements.txt
sudo pip install . 
sudo pip install . --upgrade

# For SCL's computer
#sudo pip3 install -r requirements.txt
#sudo pip3 install . 
#sudo pip3 install . --upgrade

# Execute api example
python main.py
# python3 main.py, for SCL's computer
#-------------------------------
# Remove temporary files
find . -name '*.pyc' -delete # Python2
py3clean .  # Python3

cd ..
# rm -rf RecommendationSystems # Remove any existing stuff, Not sure if able to do this
popd 

python banditApiTest.py
# python3 banditApiTest.py, for SCL's computer
